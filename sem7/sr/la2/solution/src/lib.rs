mod domain;

pub use crate::domain::*;
pub use atomic_register_public::*;
pub use register_client_public::*;
pub use sectors_manager_public::*;
pub use transfer_public::*;

mod atomic_register_public;
mod register_client_public;
mod sectors_manager_public;
mod transfer_public;
mod util;

use async_channel::unbounded;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::{
    collections::{hash_map::Entry, HashMap},
    path::PathBuf,
    sync::Arc,
};
use tokio::{
    io::{AsyncWrite, AsyncWriteExt},
    net::TcpListener,
    net::TcpStream,
    spawn,
    sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
    sync::{Mutex, Semaphore},
    task::JoinSet,
};
use util::{BoxedCmd, Callback, N_WORKERS};

pub async fn run_register_process(config: Configuration) {
    let tcp_locations = config.public.tcp_locations.clone();
    let config = Arc::new(Config::new(config));

    // we have 300ms for binding
    let listener = TcpListener::bind(config.listen_addr.clone()).await.unwrap();
    let _ = tokio::fs::create_dir_all(config.path.as_path()).await;

    let (self_tx, self_rx) = unbounded::<BoxedCmd>();
    let (ack_tx, ack_rx) = unbounded_channel::<SectorIdx>(); // for retransmissions

    let register_client = create_register_client(
        config.self_ident,
        self_tx,
        ack_rx,
        tcp_locations.clone(),
        config.hmac_system_key,
    );
    let storage_dir = config.path.clone();
    let sectors_manager = build_sectors_manager(storage_dir).await;

    let (worker_txs, worker_rxs): (
        Vec<UnboundedSender<BoxedRegisterCommand>>,
        Vec<UnboundedReceiver<BoxedRegisterCommand>>,
    ) = (0..N_WORKERS).map(|_| unbounded_channel()).unzip();

    let workers: Vec<Worker> = worker_txs
        .iter()
        .cloned()
        .map(|tx| Worker {
            semaphore: Arc::new(Semaphore::new(1)),
            tx,
        })
        .collect();

    let mut join_set = JoinSet::new();
    for rx in worker_rxs.into_iter() {
        join_set.spawn(run_worker(
            rx,
            config.clone(),
            register_client.clone(),
            sectors_manager.clone(),
        ));
    }
    join_set.spawn(async move {
        while let Ok(cmd) = self_rx.recv().await {
            let worker_id = (cmd.header.sector_idx % N_WORKERS) as usize;
            if let Some(tx) = worker_txs.get(worker_id) {
                let _ = tx.send(Box::new(WorkerCommand::System(*cmd)));
            }
        }
    });

    while let Ok((stream, _)) = listener.accept().await {
        spawn(handle_connection(
            config.clone(),
            stream,
            ack_tx.clone(),
            workers.clone(),
        ));
    }

    join_set.join_all().await;
}

struct Config {
    hmac_system_key: [u8; 64],
    hmac_client_key: [u8; 32],
    path: PathBuf,
    processes_count: u8,
    listen_addr: (String, u16),
    self_ident: u8,
    n_sectors: u64,
}

impl Config {
    fn new(config: Configuration) -> Self {
        Self {
            hmac_system_key: config.hmac_system_key,
            hmac_client_key: config.hmac_client_key,
            path: config.public.storage_dir,
            processes_count: config.public.tcp_locations.len() as u8,
            listen_addr: config
                .public
                .tcp_locations
                .get((config.public.self_rank - 1) as usize)
                .unwrap()
                .clone(),
            self_ident: config.public.self_rank,
            n_sectors: config.public.n_sectors,
        }
    }
}

#[derive(Clone)]
struct Worker {
    semaphore: Arc<Semaphore>,
    tx: UnboundedSender<BoxedRegisterCommand>,
}

type BoxedRegisterCommand = Box<WorkerCommand>;

enum WorkerCommand {
    System(SystemRegisterCommand),
    Client(ClientRegisterCommand, Callback),
}

async fn run_worker(
    mut rx: UnboundedReceiver<BoxedRegisterCommand>,
    config: Arc<Config>,
    register_client: Arc<dyn RegisterClient>,
    sectors_manager: Arc<dyn SectorsManager>,
) {
    let mut atomic_registers = HashMap::new();
    while let Some(cmd) = rx.recv().await {
        let (sector_idx, arg1, arg2);
        match *cmd {
            WorkerCommand::Client(cmd, callback) => {
                (sector_idx, arg1, arg2) = (cmd.header.sector_idx, Some((cmd, callback)), None);
            }
            WorkerCommand::System(cmd) => {
                (sector_idx, arg1, arg2) = (cmd.header.sector_idx, None, Some(cmd));
            }
        }
        if let Entry::Vacant(e) = atomic_registers.entry(sector_idx) {
            let atomic_register = build_atomic_register(
                config.self_ident,
                sector_idx,
                register_client.clone(),
                sectors_manager.clone(),
                config.processes_count,
            )
            .await;
            e.insert(atomic_register);
        }
        let atomic_register = atomic_registers.get_mut(&sector_idx).unwrap().as_mut();
        if let Some((cmd, callback)) = arg1 {
            atomic_register.client_command(cmd, callback).await;
        } else if let Some(cmd) = arg2 {
            atomic_register.system_command(cmd).await;
        }
    }
}

async fn handle_connection(
    config: Arc<Config>,
    stream: TcpStream,
    ack_tx: UnboundedSender<SectorIdx>,
    workers: Vec<Worker>,
) {
    let (mut read_stream, write_stream) = stream.into_split();
    let write_stream = Arc::new(Mutex::new(write_stream));

    while let Ok((cmd, hmac_valid)) = deserialize_register_command(
        &mut read_stream,
        &config.hmac_system_key,
        &config.hmac_client_key.clone(),
    )
    .await
    {
        handle_command(
            config.clone(),
            cmd,
            hmac_valid,
            write_stream.clone(),
            ack_tx.clone(),
            workers.clone(),
        )
        .await;
    }
}

async fn handle_command(
    config: Arc<Config>,
    cmd: RegisterCommand,
    hmac_valid: bool,
    stream: Arc<Mutex<dyn AsyncWrite + core::marker::Send + Unpin>>,
    ack_tx: UnboundedSender<SectorIdx>,
    workers: Vec<Worker>,
) {
    if hmac_valid {
        match cmd.clone() {
            RegisterCommand::Client(cmd) => {
                let worker_id = (cmd.header.sector_idx % N_WORKERS) as usize;
                if cmd.header.sector_idx < config.n_sectors {
                    let worker = &workers[worker_id];

                    let permit = worker.semaphore.clone().acquire_owned().await.unwrap();
                    let callback: Callback = Box::new(move |mut op: OperationSuccess| {
                        Box::pin(async move {
                            op.request_identifier = cmd.header.request_identifier;
                            ack_tx.send(cmd.header.sector_idx).unwrap(); // ain't no way writing to stream's gonna fail XDD
                            serialize_operation_success(
                                &op,
                                StatusCode::Ok,
                                stream,
                                &config.hmac_client_key,
                            )
                            .await;
                            drop(permit);
                        })
                    });

                    let _ = worker
                        .tx
                        .send(Box::new(WorkerCommand::Client(cmd, callback)));
                } else {
                    let op = OperationSuccess {
                        request_identifier: cmd.header.request_identifier,
                        op_return: match cmd.content {
                            ClientRegisterCommandContent::Read => {
                                OperationReturn::Read(ReadReturn {
                                    read_data: SectorVec(vec![0; 0]),
                                })
                            }
                            ClientRegisterCommandContent::Write { .. } => OperationReturn::Write,
                        },
                    };

                    serialize_operation_success(
                        &op,
                        StatusCode::InvalidSectorIndex,
                        stream.clone(),
                        &config.hmac_client_key,
                    )
                    .await;
                }
            }
            RegisterCommand::System(cmd) => {
                let worker_id = (cmd.header.sector_idx % N_WORKERS) as usize;
                let worker = &workers[worker_id];
                let _ = worker.tx.send(Box::new(WorkerCommand::System(cmd)));
            }
        }
    } else if let RegisterCommand::Client(ClientRegisterCommand { header, content }) = cmd {
        let op = OperationSuccess {
            request_identifier: header.request_identifier,
            op_return: match content {
                ClientRegisterCommandContent::Read => OperationReturn::Read(ReadReturn {
                    read_data: SectorVec(vec![0; 0]),
                }),
                ClientRegisterCommandContent::Write { .. } => OperationReturn::Write,
            },
        };

        let _ = serialize_operation_success(
            &op,
            StatusCode::AuthFailure,
            stream.clone(),
            &config.hmac_client_key,
        )
        .await;
    }
}

async fn serialize_operation_success(
    op: &OperationSuccess,
    status_code: StatusCode,
    writer: Arc<Mutex<dyn AsyncWrite + core::marker::Send + Unpin>>,
    hmac_client_key: &[u8],
) {
    let request_identifier = op.request_identifier;
    let (msg_type, mut content) = match op.clone().op_return {
        OperationReturn::Read(ReadReturn { read_data }) => (0x41_u8, read_data.0),
        OperationReturn::Write => (0x42_u8, Vec::with_capacity(0)),
    };

    let mut buf = Vec::new();

    buf.append(&mut MAGIC_NUMBER.to_vec());
    buf.append(&mut [0x00; 2].to_vec());
    buf.append(&mut [status_code as u8].to_vec());
    buf.append(&mut [msg_type].to_vec());
    buf.append(&mut request_identifier.to_be_bytes().to_vec());
    buf.append(&mut content);

    let mut mac = Hmac::<Sha256>::new_from_slice(hmac_client_key).unwrap();
    mac.update(&buf);
    let hmac_tag = mac.finalize().into_bytes();

    buf.append(&mut hmac_tag.to_vec());

    let _ = writer.lock().await.write_all(&buf).await;
}

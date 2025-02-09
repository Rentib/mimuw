use crate::{
    util::Callback, Broadcast, ClientRegisterCommand, ClientRegisterCommandContent,
    OperationReturn, OperationSuccess, ReadReturn, RegisterClient, SectorIdx, SectorVec,
    SectorsManager, SystemCommandHeader, SystemRegisterCommand, SystemRegisterCommandContent,
};
use std::{
    collections::{HashMap, HashSet},
    future::Future,
    pin::Pin,
    sync::Arc,
};
use uuid::Uuid;

#[async_trait::async_trait]
pub trait AtomicRegister: Send + Sync {
    /// Handle a client command. After the command is completed, we expect
    /// callback to be called. Note that completion of client command happens after
    /// delivery of multiple system commands to the register, as the algorithm specifies.
    ///
    /// This function corresponds to the handlers of Read and Write events in the
    /// (N,N)-AtomicRegister algorithm.
    async fn client_command(
        &mut self,
        cmd: ClientRegisterCommand,
        success_callback: Box<
            dyn FnOnce(OperationSuccess) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync,
        >,
    );

    /// Handle a system command.
    ///
    /// This function corresponds to the handlers of READ_PROC, VALUE, WRITE_PROC
    /// and ACK messages in the (N,N)-AtomicRegister algorithm.
    async fn system_command(&mut self, cmd: SystemRegisterCommand);
}

struct InternalAtomicRegister {
    self_ident: u8,
    sector_idx: SectorIdx,
    register_client: Arc<dyn RegisterClient>,
    sectors_manager: Arc<dyn SectorsManager>,
    op_id: Uuid,
    readlist: HashMap<u8, (u64, u8, SectorVec)>,
    acklist: HashSet<u8>,
    reading: bool,
    writing: bool,
    write_phase: bool,
    processes_count: u8,
    writeval: SectorVec,
    readval: SectorVec,
    callback: Option<Callback>,
}

#[async_trait::async_trait]
impl AtomicRegister for InternalAtomicRegister {
    async fn client_command(&mut self, cmd: ClientRegisterCommand, success_callback: Callback) {
        self.op_id = Uuid::new_v4();
        self.readlist.clear();
        self.acklist.clear();
        match cmd.content {
            ClientRegisterCommandContent::Read => {
                self.reading = true;
            }
            ClientRegisterCommandContent::Write { data } => {
                self.writeval = data;
                self.writing = true;
            }
        }

        self.callback = Some(success_callback);
        self.register_client
            .broadcast(Broadcast {
                cmd: Arc::new(SystemRegisterCommand {
                    header: SystemCommandHeader {
                        process_identifier: self.self_ident,
                        msg_ident: self.op_id,
                        sector_idx: self.sector_idx,
                    },
                    content: SystemRegisterCommandContent::ReadProc,
                }),
            })
            .await;
    }

    async fn system_command(&mut self, cmd: SystemRegisterCommand) {
        match cmd.content {
            SystemRegisterCommandContent::ReadProc => {
                let (ts, wr) = self
                    .sectors_manager
                    .read_metadata(cmd.header.sector_idx)
                    .await;
                let val = self.sectors_manager.read_data(cmd.header.sector_idx).await;

                self.register_client
                    .send(crate::Send {
                        cmd: Arc::new(SystemRegisterCommand {
                            header: SystemCommandHeader {
                                process_identifier: self.self_ident,
                                msg_ident: cmd.header.msg_ident,
                                sector_idx: cmd.header.sector_idx,
                            },
                            content: SystemRegisterCommandContent::Value {
                                timestamp: ts,
                                write_rank: wr,
                                sector_data: val,
                            },
                        }),
                        target: cmd.header.process_identifier,
                    })
                    .await;
            }
            SystemRegisterCommandContent::Value {
                timestamp,
                write_rank,
                sector_data,
            } if cmd.header.msg_ident == self.op_id && !self.write_phase => {
                self.readlist.insert(
                    cmd.header.process_identifier,
                    (timestamp, write_rank, sector_data),
                );

                if self.readlist.len() * 2 > self.processes_count.into()
                    && (self.reading || self.writing)
                {
                    let (ts, wr) = self
                        .sectors_manager
                        .read_metadata(cmd.header.sector_idx)
                        .await;
                    let val = self.sectors_manager.read_data(cmd.header.sector_idx).await;
                    self.readlist.insert(self.self_ident, (ts, wr, val));

                    let (maxts, rr);
                    (maxts, rr, self.readval) = self
                        .readlist
                        .iter()
                        .max_by(|a, b| a.1 .1.cmp(&b.1 .1))
                        .map(|(_, v)| v)
                        .unwrap()
                        .clone();

                    self.readlist.clear();
                    self.acklist.clear();
                    self.write_phase = true;
                    if self.reading {
                        self.register_client
                            .broadcast(Broadcast {
                                cmd: Arc::new(SystemRegisterCommand {
                                    header: SystemCommandHeader {
                                        process_identifier: self.self_ident,
                                        msg_ident: self.op_id,
                                        sector_idx: cmd.header.sector_idx,
                                    },
                                    content: SystemRegisterCommandContent::WriteProc {
                                        timestamp: maxts,
                                        write_rank: rr,
                                        data_to_write: self.readval.clone(),
                                    },
                                }),
                            })
                            .await;
                    } else {
                        let (ts, wr, val) = (maxts + 1, self.self_ident, self.writeval.clone());
                        self.sectors_manager
                            .write(cmd.header.sector_idx, &(val.clone(), ts, wr))
                            .await;
                        self.register_client
                            .broadcast(Broadcast {
                                cmd: Arc::new(SystemRegisterCommand {
                                    header: SystemCommandHeader {
                                        process_identifier: self.self_ident,
                                        msg_ident: self.op_id,
                                        sector_idx: cmd.header.sector_idx,
                                    },
                                    content: SystemRegisterCommandContent::WriteProc {
                                        timestamp: ts,
                                        write_rank: wr,
                                        data_to_write: val,
                                    },
                                }),
                            })
                            .await;
                    }
                }
            }
            SystemRegisterCommandContent::WriteProc {
                timestamp,
                write_rank,
                data_to_write,
            } => {
                let (ts, wr) = self
                    .sectors_manager
                    .read_metadata(cmd.header.sector_idx)
                    .await;

                if (timestamp, write_rank) > (ts, wr) {
                    self.sectors_manager
                        .write(
                            cmd.header.sector_idx,
                            &(data_to_write, timestamp, write_rank),
                        )
                        .await;
                }

                self.register_client
                    .send(crate::Send {
                        cmd: Arc::new(SystemRegisterCommand {
                            header: SystemCommandHeader {
                                process_identifier: self.self_ident,
                                msg_ident: cmd.header.msg_ident,
                                sector_idx: cmd.header.sector_idx,
                            },
                            content: SystemRegisterCommandContent::Ack,
                        }),
                        target: cmd.header.process_identifier,
                    })
                    .await;
            }
            SystemRegisterCommandContent::Ack
                if cmd.header.msg_ident == self.op_id && self.write_phase =>
            {
                self.acklist.insert(cmd.header.process_identifier);
                if self.acklist.len() * 2 > self.processes_count.into()
                    && (self.reading || self.writing)
                {
                    self.acklist.clear();
                    self.write_phase = false;
                    let op;
                    if self.reading {
                        self.reading = false;
                        op = Some(OperationSuccess {
                            request_identifier: 0,
                            op_return: OperationReturn::Read(ReadReturn {
                                read_data: self.readval.clone(),
                            }),
                        });
                    } else {
                        self.writing = false;
                        op = Some(OperationSuccess {
                            request_identifier: 0,
                            op_return: OperationReturn::Write,
                        });
                    }
                    if self.callback.is_some() {
                        self.callback.take().unwrap()(op.unwrap()).await;
                    }
                }
            }
            _ => {}
        }
    }
}

/// Idents are numbered starting at 1 (up to the number of processes in the system).
/// Communication with other processes of the system is to be done by register_client.
/// And sectors must be stored in the sectors_manager instance.
///
/// This function corresponds to the handlers of Init and Recovery events in the
/// (N,N)-AtomicRegister algorithm.
pub async fn build_atomic_register(
    self_ident: u8,
    sector_idx: SectorIdx,
    register_client: Arc<dyn RegisterClient>,
    sectors_manager: Arc<dyn SectorsManager>,
    processes_count: u8,
) -> Box<dyn AtomicRegister> {
    Box::new(InternalAtomicRegister {
        self_ident,
        sector_idx,
        register_client,
        sectors_manager,
        processes_count,
        op_id: Uuid::new_v4(),
        readlist: HashMap::new(),
        acklist: HashSet::new(),
        reading: false,
        writing: false,
        write_phase: false,
        writeval: SectorVec(vec![0; 4096]),
        readval: SectorVec(vec![0; 4096]),
        callback: None,
    })
}

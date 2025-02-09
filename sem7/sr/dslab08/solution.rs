use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use log::debug;
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;
use uuid::Uuid;

use module_system::{Handler, ModuleRef, System, TimerHandle};

/// A message, which disables a process. Used for testing.
pub struct Disable;

/// A message, which enables a process. Used for testing.
pub struct Enable;

struct Init;

#[derive(Clone)]
struct Timeout;

pub struct FailureDetectorModule {
    enabled: bool,
    timeout_handle: Option<TimerHandle>,
    delta: Duration,
    delay: Duration,
    // TODO add whatever fields necessary.
}

impl FailureDetectorModule {
    pub async fn new(
        system: &mut System,
        delta: Duration,
        addresses: &HashMap<Uuid, SocketAddr>,
        ident: Uuid,
    ) -> ModuleRef<Self> {
        let addr = addresses.get(&ident).unwrap();
        let socket = Arc::new(UdpSocket::bind(addr).await.unwrap());

        let module_ref = system
            .register_module(Self {
                enabled: true,
                timeout_handle: None,
                delta,
                delay: delta,
                // TODO initialize the fields you added
            })
            .await;

        tokio::spawn(deserialize_and_forward(socket, module_ref.clone()));

        module_ref.send(Init).await;

        module_ref
    }
}

#[async_trait::async_trait]
impl Handler<Init> for FailureDetectorModule {
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, _msg: Init) {
        self.timeout_handle = Some(self_ref.request_tick(Timeout, self.delay).await);
    }
}

/// New operation arrived at a socket.
#[async_trait::async_trait]
impl Handler<DetectorOperationUdp> for FailureDetectorModule {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, item: DetectorOperationUdp) {
        if self.enabled {
            unimplemented!();
        }
    }
}

/// Called periodically to check send broadcast and update alive processes.
#[async_trait::async_trait]
impl Handler<Timeout> for FailureDetectorModule {
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, _msg: Timeout) {
        if self.enabled {
            unimplemented!();
        }
    }
}

#[async_trait::async_trait]
impl Handler<Disable> for FailureDetectorModule {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, _msg: Disable) {
        self.enabled = false;
    }
}

#[async_trait::async_trait]
impl Handler<Enable> for FailureDetectorModule {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, _msg: Enable) {
        self.enabled = true;
    }
}

async fn deserialize_and_forward(
    socket: Arc<UdpSocket>,
    module_ref: ModuleRef<FailureDetectorModule>,
) {
    let mut buffer = vec![0];
    while let Ok((len, sender)) = socket.peek_from(&mut buffer).await {
        if len == buffer.len() {
            buffer.resize(2 * buffer.len(), 0);
        } else {
            socket.recv_from(&mut buffer).await.unwrap();
            match bincode::deserialize(&buffer) {
                Ok(msg) => module_ref.send(DetectorOperationUdp(msg, sender)).await,
                Err(err) => {
                    debug!("Invalid format of detector operation ({})!", err);
                }
            }
        }
    }
}

struct DetectorOperationUdp(DetectorOperation, SocketAddr);

#[derive(Serialize, Deserialize)]
pub enum DetectorOperation {
    /// Request to receive a heartbeat.
    HeartbeatRequest,
    /// Response to heartbeat, contains uuid of the receiver of HeartbeatRequest.
    HeartbeatResponse(Uuid),
    /// Request to receive information about working processes.
    AliveRequest,
    /// Vector of processes which are alive according to AliveRequest receiver.
    AliveInfo(HashSet<Uuid>),
}

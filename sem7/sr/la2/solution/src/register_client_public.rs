use crate::*;
use async_channel::{unbounded, Sender};
use std::{collections::HashMap, ops::Deref, sync::Arc};
use tokio::{
    net::TcpStream,
    select, spawn,
    sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
    time::{interval, timeout, Duration, MissedTickBehavior::Delay},
};

#[async_trait::async_trait]
/// We do not need any public implementation of this trait. It is there for use
/// in AtomicRegister. In our opinion it is a safe bet to say some structure of
/// this kind must appear in your solution.
pub trait RegisterClient: core::marker::Send + core::marker::Sync {
    /// Sends a system message to a single process.
    async fn send(&self, msg: Send);

    /// Broadcasts a system message to all processes in the system, including self.
    async fn broadcast(&self, msg: Broadcast);
}

pub struct Broadcast {
    pub cmd: Arc<SystemRegisterCommand>,
}

pub struct Send {
    pub cmd: Arc<SystemRegisterCommand>,
    /// Identifier of the target process. Those start at 1.
    pub target: u8,
}

struct InternalRegisterClient {
    self_ident: u8,
    process_count: usize,
    self_tx: Sender<BoxedCmd>,
    rebroadcast_tx: UnboundedSender<BoxedCmd>,
    txs: Vec<Sender<BoxedCmd>>,
}

impl InternalRegisterClient {
    fn new(
        self_ident: u8,
        self_tx: Sender<BoxedCmd>,
        mut ack_rx: UnboundedReceiver<SectorIdx>,
        tcp_locations: Vec<(String, u16)>,
        hmac_system_key: [u8; 64],
    ) -> Self {
        let process_count = tcp_locations.len();
        let (rebroadcast_tx, mut rebroadcast_rx): (
            UnboundedSender<BoxedCmd>,
            UnboundedReceiver<BoxedCmd>,
        ) = unbounded_channel();

        let (txs, rxs): (Vec<_>, Vec<_>) = (0..process_count).map(|_| unbounded()).unzip();

        spawn({
            // "In an actual implementation, messages should be retransmitted (with some delay
            // interval) until the sender learns that further retransmissions are guaranteed to
            // have no influence on the system’s progress."
            let txs = txs.clone();
            async move {
                let mut interval = interval(Duration::from_millis(69 * 69)); // some delay, nice
                let mut rebroadcast_cmds = HashMap::new();
                interval.set_missed_tick_behavior(Delay);
                loop {
                    select! {
                        Some(cmd) = rebroadcast_rx.recv() => {
                            rebroadcast_cmds.insert(cmd.header.sector_idx, cmd);
                        },
                        Some(sector_idx) = ack_rx.recv() => {
                            rebroadcast_cmds.remove(&sector_idx);
                        },
                        _ = interval.tick() => {
                            for ((_, cmd), tx) in rebroadcast_cmds.iter().zip(txs.iter()) {
                                let _ = tx.send(cmd.clone()).await;
                            }
                        },
                    }
                }
            }
        });

        // Yes, I'm spawning bazillion tasks like a retard instead of polling.
        tcp_locations
            .into_iter()
            .zip(rxs)
            .enumerate()
            .for_each(|(ident, (tcp_location, rx))| {
                if ident + 1 != self_ident as usize {
                    spawn(async move {
                        loop {
                            if let Ok(Ok(mut tcp_stream)) = timeout(
                                Duration::from_millis(420),
                                TcpStream::connect(tcp_location.clone()),
                            )
                            .await
                            {
                                while let Ok(cmd) = rx.recv().await {
                                    let cmd = RegisterCommand::System(cmd.deref().clone());
                                    let _ = serialize_register_command(
                                        &cmd,
                                        &mut tcp_stream,
                                        &hmac_system_key,
                                    )
                                    .await;
                                }
                            }
                        }
                    });
                }
            });

        Self {
            self_ident,
            process_count,
            self_tx,
            rebroadcast_tx,
            txs,
        }
    }
}

#[async_trait::async_trait]
impl RegisterClient for InternalRegisterClient {
    async fn send(&self, msg: Send) {
        let _ = if msg.target == self.self_ident {
            &self.self_tx
        } else {
            self.txs.get(msg.target as usize - 1).unwrap()
        }
        .send(Box::new(msg.cmd.deref().clone()))
        .await;
    }

    async fn broadcast(&self, msg: Broadcast) {
        // ensure it will get rebroadcasted until ack is received
        let _ = self.rebroadcast_tx.send(Box::new(msg.cmd.deref().clone()));
        for ident in 1..=self.process_count {
            self.send(Send {
                cmd: msg.cmd.clone(),
                target: ident as u8,
            })
            .await;
        }
    }
}

pub(crate) fn create_register_client(
    self_ident: u8,
    self_tx: Sender<BoxedCmd>,
    ack_rx: UnboundedReceiver<SectorIdx>,
    tcp_locations: Vec<(String, u16)>,
    hmac_system_key: [u8; 64],
) -> Arc<dyn RegisterClient> {
    Arc::new(InternalRegisterClient::new(
        self_ident,
        self_tx,
        ack_rx,
        tcp_locations,
        hmac_system_key,
    ))
}

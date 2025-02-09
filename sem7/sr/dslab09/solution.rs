use module_system::{Handler, ModuleRef};
use std::future::Future;
use std::pin::Pin;
use tokio::sync::oneshot::Sender;
use uuid::Uuid;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub(crate) enum ProductType {
    Electronics,
    Toys,
    Books,
}

#[derive(Clone)]
pub(crate) struct StoreMsg {
    sender: ModuleRef<DistributedStore>,
    content: StoreMsgContent,
}

#[derive(Clone, Debug)]
pub(crate) enum StoreMsgContent {
    /// Transaction Manager initiates voting for the transaction.
    RequestVote(Transaction),
    /// If every process is ok with transaction, TM issues commit.
    Commit,
    /// System-wide abort.
    Abort,
}

#[derive(Clone)]
pub(crate) struct NodeMsg {
    content: NodeMsgContent,
}

#[derive(Clone, Debug)]
pub(crate) enum NodeMsgContent {
    /// Process replies to TM whether it can/cannot commit the transaction.
    RequestVoteResponse(TwoPhaseResult),
    /// Process acknowledges to TM committing/aborting the transaction.
    FinalizationAck,
}

pub(crate) struct TransactionMessage {
    /// Request to change price.
    pub(crate) transaction: Transaction,

    /// Called after 2PC completes (i.e., the transaction was decided to be
    /// committed/aborted by DistributedStore). This must be called after responses
    /// from all processes acknowledging commit or abort are collected.
    pub(crate) completed_callback:
        Box<dyn FnOnce(TwoPhaseResult) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum TwoPhaseResult {
    Ok,
    Abort,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Product {
    pub(crate) identifier: Uuid,
    pub(crate) pr_type: ProductType,
    pub(crate) price: u64,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Transaction {
    pub(crate) pr_type: ProductType,
    pub(crate) shift: i32,
}

#[derive(Debug)]
pub(crate) struct ProductPriceQuery {
    pub(crate) product_ident: Uuid,
    pub(crate) result_sender: Sender<ProductPrice>,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ProductPrice(pub(crate) Option<u64>);

/// Message which disables a node. Used for testing.
pub(crate) struct Disable;

/// DistributedStore.
/// This structure serves as TM.
// Add any fields you need.
pub(crate) struct DistributedStore {
    nodes: Vec<ModuleRef<Node>>,
    responses: usize,
    abort: bool,
    callback:
        Option<Box<dyn FnOnce(TwoPhaseResult) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>>,
}

impl DistributedStore {
    pub(crate) fn new(nodes: Vec<ModuleRef<Node>>) -> Self {
        Self {
            nodes,
            responses: 0,
            abort: false,
            callback: None,
        }
    }

    async fn broadcast(&mut self, msg: StoreMsg) {
        for node in self.nodes.iter() {
            node.send(msg.clone()).await;
        }
    }
}

/// Node of DistributedStore.
/// This structure serves as a process of the distributed system.
// Add any fields you need.
pub(crate) struct Node {
    products: Vec<Product>,
    pending_transaction: Option<Transaction>,
    enabled: bool,
}

impl Node {
    pub(crate) fn new(products: Vec<Product>) -> Self {
        Self {
            products,
            pending_transaction: None,
            enabled: true,
        }
    }
}

#[async_trait::async_trait]
impl Handler<NodeMsg> for DistributedStore {
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, msg: NodeMsg) {
        self.responses += 1;
        match msg.content {
            NodeMsgContent::RequestVoteResponse(res) => {
                self.abort |= res == TwoPhaseResult::Abort;

                if self.responses != self.nodes.len() {
                    return;
                }

                let sender = self_ref.clone();
                let content = if self.abort {
                    StoreMsgContent::Abort
                } else {
                    StoreMsgContent::Commit
                };
                self.broadcast(StoreMsg { sender, content }).await;
            }
            NodeMsgContent::FinalizationAck => {
                if self.responses != self.nodes.len() {
                    return;
                }
                let callback = self.callback.take().unwrap();
                let res = if self.abort {
                    TwoPhaseResult::Abort
                } else {
                    TwoPhaseResult::Ok
                };
                callback(res).await;
                self.abort = false;
            }
        }
        self.responses = 0;
    }
}

#[async_trait::async_trait]
impl Handler<StoreMsg> for Node {
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, msg: StoreMsg) {
        if self.enabled {
            let sender = msg.sender.clone();
            let mut new_msg = NodeMsg {
                content: NodeMsgContent::FinalizationAck,
            };
            match msg.content {
                StoreMsgContent::RequestVote(transaction) => {
                    let products = &self.products;
                    let mut products = products.iter().filter(|p| p.pr_type == transaction.pr_type);
                    let ok = products.all(|p| {
                        if transaction.shift.is_negative() {
                            p.price > transaction.shift.wrapping_abs() as u64
                        } else {
                            p.price.checked_add(transaction.shift as u64).is_some()
                        }
                    });
                    self.pending_transaction = if ok { Some(transaction) } else { None };
                    new_msg = NodeMsg {
                        content: NodeMsgContent::RequestVoteResponse(if ok {
                            TwoPhaseResult::Ok
                        } else {
                            TwoPhaseResult::Abort
                        }),
                    };
                }
                StoreMsgContent::Commit => {
                    let pr_type = self.pending_transaction.as_ref().unwrap().pr_type;
                    let shift = self.pending_transaction.as_ref().unwrap().shift;
                    let products = self.products.iter_mut().filter(|p| p.pr_type == pr_type);
                    products.for_each(|p| {
                        if shift.is_negative() {
                            p.price -= shift.wrapping_abs() as u64;
                        } else {
                            p.price += shift as u64;
                        }
                    });
                }
                StoreMsgContent::Abort => {}
            }
            sender.send(new_msg).await;
        }
    }
}

#[async_trait::async_trait]
impl Handler<ProductPriceQuery> for Node {
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, msg: ProductPriceQuery) {
        if self.enabled {
            let res_sender = msg.result_sender;
            let _ = res_sender.send(ProductPrice(self.products.iter().find_map(|p| {
                if p.identifier == msg.product_ident {
                    Some(p.price)
                } else {
                    None
                }
            })));
        }
    }
}

#[async_trait::async_trait]
impl Handler<Disable> for Node {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, _msg: Disable) {
        self.enabled = false;
    }
}

#[async_trait::async_trait]
impl Handler<TransactionMessage> for DistributedStore {
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, msg: TransactionMessage) {
        self.callback = Some(msg.completed_callback);
        let msg = StoreMsg {
            sender: self_ref.clone(),
            content: StoreMsgContent::RequestVote(msg.transaction),
        };
        self.broadcast(msg).await;
    }
}

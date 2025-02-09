use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub trait Message: Send + 'static {}
impl<T: Send + 'static> Message for T {}

pub trait Module: Send + 'static {}
impl<T: Send + 'static> Module for T {}

/// A trait for modules capable of handling messages of type `M`.
#[async_trait::async_trait]
pub trait Handler<M: Message>: Module {
    /// Handles the message. A module must be able to access a `ModuleRef` to itself through `self_ref`.
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, msg: M);
}

/// A handle returned by `ModuleRef::request_tick()`, can be used to stop sending further ticks.
// You can add fields to this struct
pub struct TimerHandle {
    kys: Arc<AtomicBool>, // info for timer to kill itself
}

impl TimerHandle {
    /// Stops the sending of ticks resulting from the corresponding call to `ModuleRef::request_tick()`.
    /// If the ticks are already stopped, does nothing.
    pub async fn stop(&self) {
        if self.kys.load(Ordering::Relaxed) {
            return
        }
        self.kys.store(true, Ordering::Relaxed);
    }
}

// You can add fields to this struct.
pub struct System {
    kys: Arc<AtomicBool>,             // info for system to kill itself
    handles: Arc<Mutex<JoinSet<()>>>, // join handles for tokio tasks
}

impl System {
    /// Registers the module in the system.
    /// Returns a `ModuleRef`, which can be used then to send messages to the module.
    pub async fn register_module<T: Module>(&mut self, module: T) -> ModuleRef<T> {
        ModuleRef {
            kys: self.kys.clone(),
            handles: self.handles.clone(),
            module: Arc::new(Mutex::new(module)),
        }
    }

    /// Creates and starts a new instance of the system.
    pub async fn new() -> Self {
        System {
            kys: Arc::new(AtomicBool::new(false)),
            handles: Arc::new(Mutex::new(JoinSet::new())),
        }
    }

    /// Gracefully shuts the system down.
    pub async fn shutdown(&mut self) {
        self.kys.store(true, Ordering::Relaxed);
        let mut handles = self.handles.lock().await;
        while (handles.join_next().await).is_some() {}
    }
}

/// A reference to a module used for sending messages.
// You can add fields to this struct.
pub struct ModuleRef<T: Module + ?Sized> {
    // A marker field required to inform the compiler about variance in T.
    // It can be removed if type T is used in some other field.
    kys: Arc<AtomicBool>,             // info from system to kill itself
    handles: Arc<Mutex<JoinSet<()>>>, // join handles for tokio tasks
    module: Arc<Mutex<T>>,            // module itself
}

impl<T: Module> ModuleRef<T> {
    /// Sends the message to the module.
    pub async fn send<M: Message>(&self, msg: M)
    where
        T: Handler<M>,
    {
        let self_ref = self.clone();
        let kys = self_ref.kys.clone();
        if kys.load(Ordering::Relaxed) { // system is shutting down, so don't create new tasks
            return
        }
        let handles = self_ref.handles.clone();
        handles.lock().await.spawn(async move {
            let mut module = self_ref.module.lock().await;
            if kys.load(Ordering::Relaxed) {
                return
            }
            module.handle(&self_ref, msg).await;
        });
    }

    /// Schedules a message to be sent to the module periodically with the given interval.
    /// The first tick is sent after the interval elapses.
    /// Every call to this function results in sending new ticks and does not cancel
    /// ticks resulting from previous calls.
    pub async fn request_tick<M>(&self, message: M, delay: Duration) -> TimerHandle
    where
        M: Message + Clone,
        T: Handler<M>,
    {
        let tick_kys = Arc::new(AtomicBool::new(false));
        let tick_kys_clone = tick_kys.clone();
        let self_ref = self.clone();
        let system_kys = self_ref.kys.clone();

        // NOTE: this does not need to be joined
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(delay);
            interval.tick().await; // NOTE: first tick is immediate because reasons
            loop {
                interval.tick().await;
                if tick_kys.load(Ordering::Relaxed) || system_kys.load(Ordering::Relaxed) {
                   return
                }
                self_ref.send(message.clone()).await;
            }
        });

        TimerHandle { kys: tick_kys_clone }
    }
}

impl<T: Module> Clone for ModuleRef<T> {
    /// Creates a new reference to the same module.
    fn clone(&self) -> Self {
        ModuleRef {
            kys: self.kys.clone(),
            handles: self.handles.clone(),
            module: self.module.clone(),
        }
    }
}

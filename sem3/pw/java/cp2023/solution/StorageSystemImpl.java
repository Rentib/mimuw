package cp2023.solution;

import java.util.Map;

import cp2023.base.ComponentId;
import cp2023.base.DeviceId;
import cp2023.base.StorageSystem;
import cp2023.base.ComponentTransfer;
import cp2023.exceptions.*;

import java.util.concurrent.Semaphore;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.CyclicBarrier;
import java.util.ArrayList;
import java.util.HashSet;

public class StorageSystemImpl implements StorageSystem {
  private class StorageDevice {
    public final AtomicInteger slots;
    public AtomicInteger reservedSlots;
    public ConcurrentLinkedQueue<ComponentId> waiting;
    public AtomicBoolean[] reserved;
    public Semaphore[] occupiedMtx;

    public StorageDevice(Integer slots) {
      this.slots = new AtomicInteger(slots);
      this.reservedSlots = new AtomicInteger(0);
      this.waiting = new ConcurrentLinkedQueue<ComponentId>();
      this.reserved = new AtomicBoolean[slots];
      this.occupiedMtx = new Semaphore[slots];
      for (int i = 0; i < slots; i++) {
        this.reserved[i] = new AtomicBoolean(false);
        this.occupiedMtx[i] = new Semaphore(1);
      }
    }
  }

  private class StorageComponent {
    public AtomicBoolean isBeingOperatedOn;
    public Semaphore queueMtx;
    public DeviceId deviceId;
    public Integer slot;

    public StorageComponent(DeviceId deviceId, Integer slot) {
      this.isBeingOperatedOn = new AtomicBoolean(false);
      this.queueMtx = new Semaphore(0);
      this.deviceId = deviceId;
      this.slot = slot;
    }
  }

  private class CycleHandler {
    public CyclicBarrier b1;
    public CyclicBarrier b2;

    public CycleHandler(Integer cycleSize) {
      this.b1 = new CyclicBarrier(cycleSize);
      this.b2 = new CyclicBarrier(cycleSize);
    }
  }

  private ConcurrentHashMap<DeviceId, StorageDevice> devices;
  private ConcurrentHashMap<ComponentId, StorageComponent> components;
  private Semaphore globalMtx;
  private ConcurrentHashMap<ComponentId, ComponentId> cycleComponents;
  private ConcurrentHashMap<Integer, CycleHandler> cycleHandlers;
  private AtomicInteger cycleHandlerId;

  public StorageSystemImpl(
      Map<DeviceId, Integer> deviceTotalSlots,
      Map<ComponentId, DeviceId> componentPlacement
  ) {
    devices = new ConcurrentHashMap<DeviceId, StorageDevice>();
    components = new ConcurrentHashMap<ComponentId, StorageComponent>();
    Map<DeviceId, Integer> deviceCurrentSlots = new ConcurrentHashMap<DeviceId, Integer>();

    for (var entry : deviceTotalSlots.entrySet()) {
      DeviceId deviceId = entry.getKey();
      Integer slots = entry.getValue();
      devices.put(deviceId, new StorageDevice(slots));
      deviceCurrentSlots.put(deviceId, 0);
    }

    for (var entry : componentPlacement.entrySet()) {
      ComponentId componentId = entry.getKey();
      DeviceId deviceId = entry.getValue();
      Integer slot = deviceCurrentSlots.get(deviceId);

      components.put(componentId, new StorageComponent(deviceId, slot));

      StorageDevice device = devices.get(deviceId);
      device.reserved[slot].set(true);
      device.reservedSlots.incrementAndGet();
      mtxLock(device.occupiedMtx[slot]);

      deviceCurrentSlots.replace(deviceId, slot + 1);
    }

    globalMtx = new Semaphore(1);
    cycleComponents = new ConcurrentHashMap<ComponentId, ComponentId>();

    cycleHandlers = new ConcurrentHashMap<Integer, CycleHandler>();
    cycleHandlerId = new AtomicInteger(0);
  }

  private void mtxLock(Semaphore mtx) {
    try {
      mtx.acquire();
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private void mtxUnlock(Semaphore mtx) {
    mtx.release();
  }

  private void checkTransfer(ComponentTransfer transfer) throws TransferException {
    if (transfer.getSourceDeviceId() == null && transfer.getDestinationDeviceId() == null)
      throw new IllegalTransferType(transfer.getComponentId());

    if (transfer.getSourceDeviceId() != null
    && !devices.containsKey(transfer.getSourceDeviceId()))
      throw new DeviceDoesNotExist(transfer.getSourceDeviceId());
    if (transfer.getDestinationDeviceId() != null
    && !devices.containsKey(transfer.getDestinationDeviceId()))
      throw new DeviceDoesNotExist(transfer.getDestinationDeviceId());

    if (transfer.getSourceDeviceId() == null
    &&  components.containsKey(transfer.getComponentId())
    &&  components.get(transfer.getComponentId()).deviceId == null)
      throw new ComponentAlreadyExists(transfer.getComponentId());

    if (transfer.getSourceDeviceId() == null
    &&  components.containsKey(transfer.getComponentId())
    &&  components.get(transfer.getComponentId()).deviceId != null)
      throw new ComponentAlreadyExists(transfer.getComponentId(), components.get(transfer.getComponentId()).deviceId);

    if (transfer.getSourceDeviceId() != null) {
      if (!components.containsKey(transfer.getComponentId()))
        throw new ComponentDoesNotExist(transfer.getComponentId(), transfer.getSourceDeviceId());
      if (!components.get(transfer.getComponentId()).deviceId.equals(transfer.getSourceDeviceId()))
        throw new ComponentDoesNotExist(transfer.getComponentId(), transfer.getSourceDeviceId());
    }

    if (transfer.getSourceDeviceId() != null
    &&  transfer.getDestinationDeviceId() != null
    &&  transfer.getSourceDeviceId().equals(transfer.getDestinationDeviceId()))
      throw new ComponentDoesNotNeedTransfer(transfer.getComponentId(), transfer.getSourceDeviceId());
    
    if (components.containsKey(transfer.getComponentId())
    &&  components.get(transfer.getComponentId()).isBeingOperatedOn.get())
      throw new ComponentIsBeingOperatedOn(transfer.getComponentId());
  }

  @Override
  public void execute(ComponentTransfer transfer) throws TransferException {
    mtxLock(globalMtx);
    
    try {
      checkTransfer(transfer);
    } catch (TransferException e) {
      mtxUnlock(globalMtx);
      throw e;
    }

    if (transfer.getSourceDeviceId() != null && transfer.getDestinationDeviceId() != null)
      move(transfer);
    else if (transfer.getSourceDeviceId() != null)
      remove(transfer);
    else if (transfer.getDestinationDeviceId() != null)
      add(transfer);
  }

  private void add(ComponentTransfer transfer) {
    components.put(transfer.getComponentId(), new StorageComponent(null, null));
    StorageComponent component = components.get(transfer.getComponentId());
    StorageDevice dst = devices.get(transfer.getDestinationDeviceId());

    component.isBeingOperatedOn.set(true);

    reserveSlot(transfer, component, dst);

    transfer.prepare();

    mtxLock(dst.occupiedMtx[component.slot]);
    component.deviceId = transfer.getDestinationDeviceId();
    
    transfer.perform();

    mtxLock(globalMtx);
    component.isBeingOperatedOn.set(false);
    mtxUnlock(globalMtx);
  }

  private void remove(ComponentTransfer transfer) {
    StorageComponent component = components.get(transfer.getComponentId());
    StorageDevice src = devices.get(transfer.getSourceDeviceId());
    
    component.isBeingOperatedOn.set(true);
    mtxUnlock(globalMtx);

    transfer.prepare();

    src.reserved[component.slot].set(false);
    src.reservedSlots.decrementAndGet();
    if (!src.waiting.isEmpty()) {
      StorageComponent first = components.get(src.waiting.peek());
      mtxUnlock(first.queueMtx);
    }

    mtxUnlock(src.occupiedMtx[component.slot]);

    transfer.perform();

    mtxLock(globalMtx);
    component.isBeingOperatedOn.set(false);
    components.remove(transfer.getComponentId());
    mtxUnlock(globalMtx);
  }

  private void move(ComponentTransfer transfer) {
    StorageComponent component = components.get(transfer.getComponentId());
    StorageDevice src = devices.get(transfer.getSourceDeviceId());
    StorageDevice dst = devices.get(transfer.getDestinationDeviceId());

    component.isBeingOperatedOn.set(true);

    if (handleCycle(transfer))
      return;

    Integer oldSlot = component.slot;
    if (reserveSlot(transfer, component, dst))
      return;

    transfer.prepare();

    src.reserved[oldSlot].set(false);
    src.reservedSlots.decrementAndGet();
    if (!src.waiting.isEmpty()) {
      StorageComponent first = components.get(src.waiting.peek());
      mtxUnlock(first.queueMtx);
    }

    mtxLock(dst.occupiedMtx[component.slot]);
    component.deviceId = transfer.getDestinationDeviceId();
    mtxUnlock(src.occupiedMtx[oldSlot]);

    transfer.perform();

    mtxLock(globalMtx);
    component.isBeingOperatedOn.set(false);
    mtxUnlock(globalMtx);
  }

  private boolean reserveSlot(ComponentTransfer transfer, StorageComponent component, StorageDevice device) {
    while (true) {
      if (device.slots.get() > device.reservedSlots.get()) {
        for (int i = 0; i < device.slots.get(); i++) {
          if (device.reserved[i].get())
            continue;
          component.slot = i;
          device.reserved[i].set(true);
          device.reservedSlots.incrementAndGet();
          mtxUnlock(globalMtx);
          return false;
        }
      } else {
        device.waiting.add(transfer.getComponentId());
        mtxUnlock(globalMtx);
        mtxLock(component.queueMtx);
        device.waiting.remove(transfer.getComponentId());

        if (cycleComponents.contains(transfer.getComponentId())) {
          CycleHandler cycleHandler = cycleHandlers.get(cycleHandlerId.get());
          var next = cycleComponents.get(transfer.getComponentId());
          if (next != transfer.getComponentId())
            mtxUnlock(components.get(next).queueMtx);

          cycleComponents.remove(transfer.getComponentId());

          try {
            cycleHandler.b1.await();
          } catch (Exception e) {
            throw new RuntimeException(e);
          }

          transfer.prepare();

          try {
            cycleHandler.b2.await();
          } catch (Exception e) {
            throw new RuntimeException(e);
          }

          transfer.perform();

          components.get(transfer.getComponentId()).isBeingOperatedOn.set(false);

          return true;
        } else {
          mtxLock(globalMtx);
        }
      }
    }
  }

  private ArrayList<ComponentId> cycle;
  private HashSet<ComponentId> visited;
  private boolean handleCycle(ComponentTransfer transfer) {
    cycle = new ArrayList<ComponentId>();
    visited = new HashSet<ComponentId>();

    if (!findCycle(transfer.getComponentId(), transfer.getDestinationDeviceId()))
      return false;

    Integer id = cycleHandlerId.incrementAndGet();
    cycleHandlers.put(id, new CycleHandler(cycle.size()));
    CycleHandler cycleHandler = cycleHandlers.get(id);

    cycleComponents.clear();
    for (int i = 1; i < cycle.size() - 1; i++)
      cycleComponents.put(cycle.get(i), cycle.get(i + 1));
    cycleComponents.put(cycle.get(cycle.size() - 1), cycle.get(cycle.size() - 1));

    var cycleDevices = new ArrayList<DeviceId>();
    var cycleSlots = new ArrayList<Integer>();
    for (int i = 0; i < cycle.size(); i++) {
      cycleDevices.add(components.get(cycle.get(i)).deviceId);
      cycleSlots.add(components.get(cycle.get(i)).slot);
    }

    for (int i = 0; i < cycle.size(); i++) {
      StorageComponent component = components.get(cycle.get(i));
      int target = (i - 1 + cycle.size()) % cycle.size();

      component.deviceId = cycleDevices.get(target);
      component.slot = cycleSlots.get(target);
    }

    mtxUnlock(components.get(cycle.get(1)).queueMtx);
    
    cycleComponents.remove(transfer.getComponentId());

    try {
      cycleHandler.b1.await();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    mtxUnlock(globalMtx);

    transfer.prepare();

    try {
      cycleHandler.b2.await();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    transfer.perform();

    mtxLock(globalMtx);
    components.get(transfer.getComponentId()).isBeingOperatedOn.set(false);
    mtxUnlock(globalMtx);

    cycleHandlers.remove(id);

    return true;
  }

  private boolean findCycle(ComponentId v, DeviceId dst) {
    if (visited.contains(v))
      return false;
    visited.add(v);

    if (components.get(v).deviceId == null) // component being added not moved
      return false;

    cycle.add(v);

    if (components.get(v).deviceId.equals(dst)) // cycle found
      return true;

    for (ComponentId u : devices.get(components.get(v).deviceId).waiting) {
      if (findCycle(u, dst))
        return true;
    }
    cycle.remove(cycle.size() - 1);
    return false;
  }
}

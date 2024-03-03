/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.demo;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import cp2023.base.ComponentId;
import cp2023.base.ComponentTransfer;
import cp2023.base.DeviceId;
import cp2023.base.StorageSystem;
import cp2023.exceptions.TransferException;
import cp2023.solution.StorageSystemFactory;


public final class TransferBurst {

    public static void main(String[] args) {
        StorageSystem system = setupSystem();
        Collection<Thread> users = setupTransferers(system);
        runTransferers(users);
    }

    private final static StorageSystem setupSystem() {
        DeviceId dev1 = new DeviceId(1);
        DeviceId dev2 = new DeviceId(2);
        DeviceId dev3 = new DeviceId(3);
        
        ComponentId comp1 = new ComponentId(101);
        ComponentId comp2 = new ComponentId(102);
        ComponentId comp3 = new ComponentId(103);
        ComponentId comp4 = new ComponentId(104);
        ComponentId comp5 = new ComponentId(105);
        ComponentId comp6 = new ComponentId(106);
        ComponentId comp7 = new ComponentId(107);
        ComponentId comp8 = new ComponentId(108);
        ComponentId comp9 = new ComponentId(109);
        
        HashMap<DeviceId, Integer> deviceCapacities = new HashMap<>(3);
        deviceCapacities.put(dev1, 3);
        deviceCapacities.put(dev2, 3);
        deviceCapacities.put(dev3, 5);
        
        HashMap<ComponentId, DeviceId> initialComponentMapping = new HashMap<>(9);

        initialComponentMapping.put(comp1, dev1);
        initialComponentMapping.put(comp2, dev1);
        initialComponentMapping.put(comp3, dev1);
        
        initialComponentMapping.put(comp4, dev2);
        initialComponentMapping.put(comp5, dev2);
        initialComponentMapping.put(comp6, dev2);
        
        initialComponentMapping.put(comp7, dev3);
        initialComponentMapping.put(comp8, dev3);
        initialComponentMapping.put(comp9, dev3);
        
        return StorageSystemFactory.newSystem(deviceCapacities, initialComponentMapping);
    }
    
    private final static Collection<Thread> setupTransferers(StorageSystem system) {
        // FIXME: debug
        ArrayList<Thread> transferer = new ArrayList<>();
        transferer.add(new Thread(new Runnable() {
            @Override
            public void run() {
                sleep(10);
                System.out.println("Transferer " + Thread.currentThread().getId() + " has started.");
                executeTransfer(system, 101, 1, 3, 20);
                sleep(30);
                executeTransfer(system, 105, 2, 0, 10);
                System.out.println("Transferer " + Thread.currentThread().getId() + " has finished.");
            }
        }));
        transferer.add(new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("Transferer " + Thread.currentThread().getId() + " has started.");
                executeTransfer(system, 110, 0, 1, 10);
                System.out.println("Transferer " + Thread.currentThread().getId() + " has finished.");
            }
        }));
        transferer.add(new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("Transferer " + Thread.currentThread().getId() + " has started.");
                executeTransfer(system, 109, 3, 2, 10);
                sleep(30);
                executeTransfer(system, 102, 1, 0, 10);
                System.out.println("Transferer " + Thread.currentThread().getId() + " has finished.");
            }
        }));
        transferer.add(new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("Transferer " + Thread.currentThread().getId() + " has started.");
                executeTransfer(system, 107, 3, 1, 10);
                System.out.println("Transferer " + Thread.currentThread().getId() + " has finished.");
            }
        }));
        return transferer;
    }
    
    private final static void runTransferers(Collection<Thread> users) {
        for (Thread t : users) {
            t.start();
        }
        for (Thread t : users) {
            try {
                t.join();
            } catch (InterruptedException e) {
                throw new RuntimeException("panic: unexpected thread interruption", e);
            }
        }
    }

    
    
    private final static CompTransfImpl executeTransfer(
            StorageSystem system,
            int compId,
            int srcDevId,
            int dstDevId,
            long duration
    ) {
        CompTransfImpl transfer =
                new CompTransfImpl(
                        new ComponentId(compId),
                        srcDevId > 0 ? new DeviceId(srcDevId) : null,
                        dstDevId > 0 ? new DeviceId(dstDevId) : null,
                        duration
                );
        try {
            system.execute(transfer);
        } catch (TransferException e) {
            throw new RuntimeException("Uexpected transfer exception: " + e.toString(), e);
        }
        return transfer;
    }
    
    private final static void sleep(long duration) {
        try {
            Thread.sleep(duration);
        } catch (InterruptedException e) {
            throw new RuntimeException("panic: unexpected thread interruption", e);
        }
    }
    

    
    private final static class CompTransfImpl implements ComponentTransfer {
        private static int uidGenerator = 0;
        private final int uid;
        private final long owningThread;
        private final Integer phantomSynchronizer;
        private final ComponentId compId;
        private final DeviceId srcDevId;
        private final DeviceId dstDevId;
        private final long duration;
        private boolean prepared;
        private boolean started;
        private boolean done;
        
        private final static synchronized int generateUID() {
            return ++uidGenerator;
        }
        
        public CompTransfImpl(
                ComponentId compId,
                DeviceId srcDevId,
                DeviceId dstDevId,
                long duration
        ) {
            this.uid = generateUID();
            this.phantomSynchronizer = 19;
            this.owningThread = Thread.currentThread().getId();
            this.compId = compId;
            this.srcDevId = srcDevId;
            this.dstDevId = dstDevId;
            this.duration = duration;
            this.prepared = false;
            this.started = false;
            this.done = false;
            System.out.println("Transferer " + this.owningThread +
                    " is about to issue transfer " + this.uid +
                    " of " + this.compId + " from " + this.srcDevId +
                    " to " + this.dstDevId + ".");
        }
        
        @Override
        public ComponentId getComponentId() {
            return this.compId;
        }

        @Override
        public DeviceId getSourceDeviceId() {
            return this.srcDevId;
        }

        @Override
        public DeviceId getDestinationDeviceId() {
            return this.dstDevId;
        }

        @Override
        public void prepare() {
            synchronized (this.phantomSynchronizer) {
                if (this.prepared) {
                    throw new RuntimeException(
                            "Transfer " + this.uid + " is being prepared more than once!");
                }
                if (this.owningThread != Thread.currentThread().getId()) {
                    throw new RuntimeException(
                            "Transfer " + this.uid +
                            " is being prepared by a different thread that scheduled it!");
                }
                this.prepared = true;
            }
            System.out.println("Transfer " + this.uid + " of " + this.compId +
                    " from " + this.srcDevId + " to " + this.dstDevId +
                    " has been prepared by user " + Thread.currentThread().getId() + ".");
        }

        @Override
        public void perform() {
            synchronized (this.phantomSynchronizer) {
                if (! this.prepared) {
                    throw new RuntimeException(
                            "Transfer " + this.uid + " has not been prepared " +
                            "before being performed!");
                }
                if (this.started) {
                    throw new RuntimeException(
                            "Transfer " + this.uid + " is being started more than once!");
                }
                if (this.owningThread != Thread.currentThread().getId()) {
                    throw new RuntimeException(
                            "Transfer " + this.uid +
                            " is being performed by a different thread that scheduled it!");
                }
                this.started = true;
            }
            System.out.println("Transfer " + this.uid + " of " + this.compId +
                    " from " + this.srcDevId + " to " + this.dstDevId + " has been started.");
            sleep(this.duration);
            synchronized (this.phantomSynchronizer) {
                this.done = true;
            }
            System.out.println("Transfer " + this.uid + " of " + this.compId +
                    " from " + this.srcDevId + " to " + this.dstDevId + " has been completed.");
        }
        
    }
    
}

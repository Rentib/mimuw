/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.exceptions;

import cp2023.base.ComponentId;
import cp2023.base.DeviceId;

public final class ComponentDoesNotNeedTransfer extends TransferException {

    private static final long serialVersionUID = -8381504188759878376L;

    private final ComponentId compId;
    private final DeviceId    devId;
    
    public ComponentDoesNotNeedTransfer(ComponentId compId, DeviceId devId) {
        super("component " + compId.toString() +
                " does not need a transfer from device " + devId.toString() +
                " to the same device");
        this.compId = compId;
        this.devId = devId;
    }
    
    public ComponentId getComponentId() {
        return this.compId;
    }

    public DeviceId getDeviceId() {
        return this.devId;
    }
}

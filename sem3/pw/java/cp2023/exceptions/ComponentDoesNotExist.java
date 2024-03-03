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

public final class ComponentDoesNotExist extends TransferException {

    private static final long serialVersionUID = -6602924098139024228L;

    private final ComponentId compId;
    private final DeviceId    devId;
    
    public ComponentDoesNotExist(ComponentId compId, DeviceId devId) {
        super("component " + compId.toString() + " does not exist on device " + devId.toString());
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

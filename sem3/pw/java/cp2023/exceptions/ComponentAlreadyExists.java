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

public final class ComponentAlreadyExists extends TransferException {

    private static final long serialVersionUID = -2454850802956930740L;

    private final ComponentId compId;
    private final DeviceId    devId;
    
    public ComponentAlreadyExists(ComponentId compId) {
        super("component " + compId.toString() + " already awaits to be uploaded");
        this.compId = compId;
        this.devId = null;
    }
    
    public ComponentAlreadyExists(ComponentId compId, DeviceId devId) {
        super("component " + compId.toString() + " already exists on device " + devId.toString());
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

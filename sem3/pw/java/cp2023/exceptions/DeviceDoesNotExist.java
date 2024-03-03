/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.exceptions;

import cp2023.base.DeviceId;

public final class DeviceDoesNotExist extends TransferException {

    private static final long serialVersionUID = -4725693092228884970L;

    private final DeviceId devId;
    
    public DeviceDoesNotExist(DeviceId devId) {
        super("device " + devId.toString() + " does not exist");
        this.devId = devId;
    }
    
    public DeviceId getDeviceId() {
        return this.devId;
    }
}

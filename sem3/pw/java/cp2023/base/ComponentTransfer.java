/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.base;

public interface ComponentTransfer {

    public ComponentId getComponentId();
    
    public DeviceId getSourceDeviceId();
    
    public DeviceId getDestinationDeviceId();
    
    public void prepare();
    
    public void perform();
}

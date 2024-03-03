/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.exceptions;

import cp2023.base.ComponentId;

public final class ComponentIsBeingOperatedOn extends TransferException {

    private static final long serialVersionUID = 709346310931435664L;

    private final ComponentId compId;
    
    public ComponentIsBeingOperatedOn(ComponentId compId) {
        super("component " + compId.toString() + " is being operated on");
        this.compId = compId;
    }
    
    public ComponentId getComponentId() {
        return this.compId;
    }
}

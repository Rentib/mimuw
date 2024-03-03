/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.exceptions;

import cp2023.base.ComponentId;

public final class IllegalTransferType extends TransferException {

    private static final long serialVersionUID = 2160512945697448653L;

    private final ComponentId compId;
    
    public IllegalTransferType(ComponentId compId) {
        super("both source and destination devices are null " +
                "for component " + compId.toString());
        this.compId = compId;
    }
    
    public ComponentId getComponentId() {
        return this.compId;
    }
}

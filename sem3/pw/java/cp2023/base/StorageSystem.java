/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.base;

import cp2023.exceptions.TransferException;

public interface StorageSystem {

    void execute(ComponentTransfer transfer) throws TransferException;
    
}

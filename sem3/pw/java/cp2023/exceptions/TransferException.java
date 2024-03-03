/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.exceptions;

public abstract class TransferException extends Exception {

    private static final long serialVersionUID = -4456854647932628439L;

    public TransferException(String message) {
        super(message);
    }
}

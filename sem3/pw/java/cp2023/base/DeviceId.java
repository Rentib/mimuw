/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.base;

public final class DeviceId implements Comparable<DeviceId> {

    private int id;
    
    public DeviceId(int id) {
        this.id = id;
    }

    @Override
    public boolean equals(Object obj) {
        if (! (obj instanceof DeviceId)) {
            return false;
        }
        return this.id == ((DeviceId)obj).id;
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(this.id);
    }

    @Override
    public String toString() {
        return "DEV-" + this.id;
    }

    @Override
    public int compareTo(DeviceId other) {
        return Integer.compare(this.id, other.id);
    }
    
    
}

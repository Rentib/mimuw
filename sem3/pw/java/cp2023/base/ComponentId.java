/*
 * University of Warsaw
 * Concurrent Programming Course 2023/2024
 * Java Assignment
 *
 * Author: Konrad Iwanicki (iwanicki@mimuw.edu.pl)
 */
package cp2023.base;

public final class ComponentId implements Comparable<ComponentId> {

    private int id;
    
    public ComponentId(int id) {
        this.id = id;
    }

    @Override
    public boolean equals(Object obj) {
        if (! (obj instanceof ComponentId)) {
            return false;
        }
        return this.id == ((ComponentId)obj).id;
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(this.id);
    }

    @Override
    public String toString() {
        return "COMP-" + this.id;
    }

    @Override
    public int compareTo(ComponentId other) {
        return Integer.compare(this.id, other.id);
    }
    
    
}

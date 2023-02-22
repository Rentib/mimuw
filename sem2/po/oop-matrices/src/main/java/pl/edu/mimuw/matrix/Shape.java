package pl.edu.mimuw.matrix;

import java.util.Objects;

public final class Shape {
    public final int rows;
    public final int columns;
    private Shape(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
    }
    void assertInShape(int row, int column) {
        assert row >= 0;
        assert row < rows;
        assert column >= 0;
        assert column < columns;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Shape shape = (Shape) o;
        return rows == shape.rows && columns == shape.columns;
    }
    @Override
    public int hashCode() {
        return Objects.hash(rows, columns);
    }
    public static Shape vector(int size) {
        return Shape.matrix(size, 1);
    }
    public static Shape matrix(int rows, int columns) {
        assert columns > 0;
        assert rows > 0;
        return new Shape(rows, columns);
    }
}


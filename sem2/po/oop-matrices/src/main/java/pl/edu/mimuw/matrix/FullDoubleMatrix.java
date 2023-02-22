package pl.edu.mimuw.matrix;

public class FullDoubleMatrix extends DoubleMatrix {
    private double[][] v;
    public FullDoubleMatrix(double[][] xd) {
        assert(xd != null && xd.length > 0 && xd[0].length > 0);
        shape = Shape.matrix(xd.length, xd[0].length);
        v = new double[shape.rows][shape.columns];
        for (int row = 0; row < shape.rows; row++) {
            assert(xd[0].length == xd[row].length);
            for (int column = 0; column < shape.columns; column++)
                v[row][column] = xd[row][column];
        }
    }
    public static IDoubleMatrix mult(IDoubleMatrix a, IDoubleMatrix b) {
        assert(a.shape().rows == b.shape().columns);
        if (a.getClass() == IdentityDoubleMatrix.class) return b;
        if (b.getClass() == IdentityDoubleMatrix.class) return a;
        if (a.getClass() == ConstDoubleMatrix.class && a.get(0, 0) == 0
        ||  b.getClass() == ConstDoubleMatrix.class && b.get(0, 0) == 0)
            return new ConstDoubleMatrix(Shape.matrix(a.shape().rows, b.shape().columns), 0);
        double[][] xd = new double[a.shape().rows][b.shape().columns];
        for (int row = 0; row < a.shape().rows; row++) {
            for (int column = 0; column < b.shape().columns; column++) {
                xd[row][column] = 0;
                for (int i = 0; i < b.shape().rows; i++)
                    xd[row][column] += a.get(row, i) * b.get(i, column);
            }
        }
        return new FullDoubleMatrix(xd);
    }
    public static IDoubleMatrix add(IDoubleMatrix a, IDoubleMatrix b) {
        assert(a.shape().equals(b.shape()));
        if (a.getClass() == ConstDoubleMatrix.class && a.get(0, 0) == 0) return b;
        if (b.getClass() == ConstDoubleMatrix.class && b.get(0, 0) == 0) return a;
        double[][] xd = new double[a.shape().rows][a.shape().columns];
        for (int row = 0; row < a.shape().rows; row++)
            for (int column = 0; column < a.shape().columns; column++)
                xd[row][column] = a.get(row, column) + b.get(row, column);
        return new FullDoubleMatrix(xd);
    }
    public IDoubleMatrix times(IDoubleMatrix other) {
        return mult(this, other);
    }
    public IDoubleMatrix times(double scalar) {
        if (scalar == 0) return new ConstDoubleMatrix(shape, 0);
        if (scalar == 1) return this;
        double[][] xd = new double[shape.rows][shape.columns];
        for (int row = 0; row < shape.rows; row++)
            for (int column = 0; column < shape.columns; column++)
                xd[row][column] = this.get(row, column) * scalar;
        return new FullDoubleMatrix(xd);
    }
    public IDoubleMatrix plus(IDoubleMatrix other) {
        return add(this, other);
    }
    public IDoubleMatrix plus(double scalar) {
        double[][] xd = new double[shape.rows][shape.columns];
        for (int row = 0; row < shape.rows; row++)
            for (int column = 0; column < shape.columns; column++)
                xd[row][column] = this.get(row, column) + scalar;
        return new FullDoubleMatrix(xd);
    }
    public double get(int row, int column) {
        shape.assertInShape(row, column);
        return v[row][column];
    }
    public double normOne() {
        double res = 0;
        for (int column = 0, value = 0; column < shape.columns;
             column++, res = Math.max(res, value), value = 0)
            for (int row = 0; row < shape.rows;
                 value += Math.abs(this.get(row++, column)));
        return res;
    }
    public double normInfinity() {
        double res = 0;
        for (int row = 0, value = 0; row < shape.rows;
             row++, res = Math.max(res, value), value = 0)
            for (int column = 0; column < shape.columns;
                 value += Math.abs(this.get(row, column++)));
        return res;
    }
    public double frobeniusNorm() {
        double res = 0;
        for (int row = 0; row < shape.rows; row++)
            for (int column = 0; column < shape.columns;
                res += this.get(row, column) * this.get(row, column++));
        return Math.sqrt(res);
    }
    @Override public String toString() {
        StringBuilder s = new StringBuilder("Type: full\nShape: " +
                shape.rows + " x " + shape.columns + "\n");
        for (int row = 0; row < shape.rows; row++, s.append("\n")) {
            for (int column = 0; column < shape.columns; column++, s.append(" ")) {
                if (0 < column && column < shape.columns - 1)
                    s.append((this.get(row, column - 1) == this.get(row, column)
                            && this.get(row, column) == this.get(row, column + 1)
                            ? "..." : this.get(row, column)));
                else s.append(this.get(row, column));
            }
        }
        return s.toString();
    }
}


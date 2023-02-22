package pl.edu.mimuw.matrix;

public abstract class DoubleMatrix implements IDoubleMatrix {
    protected Shape shape;
    abstract public IDoubleMatrix times(IDoubleMatrix other);
    abstract public IDoubleMatrix times(double scalar);
    abstract public IDoubleMatrix plus(IDoubleMatrix other);
    abstract public IDoubleMatrix plus(double scalar);
    public IDoubleMatrix minus(IDoubleMatrix other) {
        return this.plus(other.times(-1));
    }
    public IDoubleMatrix minus(double scalar) { return this.plus(-scalar); }
    abstract public double get(int row, int column);
    public double[][] data() {
        double[][] xd = new double[shape.rows][shape.columns];
        for (int row = 0; row < shape.rows; row++)
            for (int column = 0; column < shape.columns; column++)
                xd[row][column] = this.get(row, column);
        return xd;
    }
    abstract public double normOne();
    abstract public double normInfinity();
    abstract public double frobeniusNorm();
    abstract public String toString();
    public Shape shape() { return shape; }
}


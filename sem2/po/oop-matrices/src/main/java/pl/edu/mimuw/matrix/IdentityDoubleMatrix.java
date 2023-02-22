package pl.edu.mimuw.matrix;

class IdentityDoubleMatrix extends DoubleMatrix {
    public IdentityDoubleMatrix(int size) { shape = Shape.matrix(size, size); }
    public IDoubleMatrix times(IDoubleMatrix other) {
        return FullDoubleMatrix.mult(other, this);
    }
    public IDoubleMatrix times(double scalar) {
        if (scalar == 1) return this;
        double[] xd = new double[shape.rows];
        for (int i = 0; i < xd.length; i++) xd[i] = scalar;
        return new DiagonalDoubleMatrix(xd);
    }
    public IDoubleMatrix plus(IDoubleMatrix other) {
        return FullDoubleMatrix.add(this, other);
    }
    public IDoubleMatrix plus(double scalar) {
        return FullDoubleMatrix.add(this, new ConstDoubleMatrix(shape, scalar));
    }
    public double get(int row, int column) {
        shape.assertInShape(row, column);
        return row == column ? 1 : 0;
    }
    public double normOne() { return 1; }
    public double normInfinity() { return 1; }
    public double frobeniusNorm() { return Math.sqrt(shape.rows); }
    @Override public String toString() {
        return "Type: identity\nShape: " + shape.rows +" x " + shape.columns + "\n";
    }
}


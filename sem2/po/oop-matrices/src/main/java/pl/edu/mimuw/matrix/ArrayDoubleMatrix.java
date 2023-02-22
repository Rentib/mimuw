package pl.edu.mimuw.matrix;

abstract class ArrayDoubleMatrix extends DoubleMatrix {
    protected double[] array;
    public IDoubleMatrix plus(double scalar) {
        return FullDoubleMatrix.add(this, new ConstDoubleMatrix(shape, scalar));
    }
    public double normOne() {
        double res = 0;
        for (var i : array) res = Math.max(res, Math.abs(i));
        return res;
    }
    public double normInfinity() { return this.normOne(); }
    public double frobeniusNorm() {
        double res = 0;
        for (var i : array) res += i * i;
        return Math.sqrt(res);
    }
}


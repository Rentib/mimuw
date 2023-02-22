package pl.edu.mimuw.matrix;

public class SameRowsDoubleMatrix extends ArrayDoubleMatrix {
    public SameRowsDoubleMatrix(Shape _shape, double ...v) {
        shape = _shape;
        array = v.clone();
    }
    @Override public IDoubleMatrix times(IDoubleMatrix other) {
        return (new FullDoubleMatrix(this.data())).times(other);
    }
    @Override public IDoubleMatrix times(double scalar) {
        double[] xd = array.clone();
        for (int i = 0; i < xd.length; i++) xd[i] *= scalar;
        return new SameRowsDoubleMatrix(shape, xd);
    }
    @Override public IDoubleMatrix plus(IDoubleMatrix other) {
        assert(shape == other.shape());
        if (other.getClass() == SameRowsDoubleMatrix.class) {
            double[] xd = array.clone();
            for (int i = 0; i < xd.length; i++)
                xd[i] += other.get(0, i);
        }
        return (new FullDoubleMatrix(this.data())).plus(other);
    }
    @Override public IDoubleMatrix plus(double scalar) {
        double[] xd = array.clone();
        for (int i = 0; i < xd.length; i++) xd[i] += scalar;
        return new SameRowsDoubleMatrix(shape, xd);
    }
    @Override public double get(int row, int column) {
        shape.assertInShape(row, column);
        return array[column];
    }
    @Override public double normOne() {
        double res = 0;
        for (var i : array)
            res = Math.max(res, Math.abs(i) * shape.rows);
        return res;
    }
    @Override public double normInfinity() {
        double res = 0;
        for (var i : array)
            res += Math.abs(i);
        return res;
    }
    @Override  public double frobeniusNorm() {
        double res = 0;
        for (var i : array)
            res += i * i * shape.rows;
        return res;
    }
    @Override public String toString() {
        StringBuilder s = new StringBuilder("Type: samerows\nShape: " + shape.rows + "x" + shape.columns + "\n");
        StringBuilder z = new StringBuilder("");
        for (var i : array) z.append(i + " ");
        for (int i = 0;i < shape.rows;i++) s.append(z + "\n");
        return s.toString();
    }
}

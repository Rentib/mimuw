package pl.edu.mimuw.matrix;

public class DiagonalDoubleMatrix extends ArrayDoubleMatrix {
    public DiagonalDoubleMatrix(double ...xd) {
        assert(xd.length > 0);
        shape = Shape.matrix(xd.length, xd.length);
        array = xd.clone();
    }
    public IDoubleMatrix times(IDoubleMatrix other) {
        assert(this.shape().rows == other.shape().columns);
        if (other.getClass() == IdentityDoubleMatrix.class) return this;
        if (this.getClass() == other.getClass()) {
            double[] xd = array.clone();
            for (int i = 0; i < array.length; i++) xd[i] *= other.get(i, i);
            return new DiagonalDoubleMatrix(xd);
        }
        return FullDoubleMatrix.mult(this, other);
    }
    public IDoubleMatrix times(double scalar) {
        if (scalar == 0) return new ConstDoubleMatrix(shape, 0);
        if (scalar == 1) return this;
        double[] xd = array.clone();
        for (int i = 0; i < array.length; xd[i++] *= scalar);
        return new DiagonalDoubleMatrix(xd);
    }
    public IDoubleMatrix plus(IDoubleMatrix other) {
        assert(shape.equals(other.shape()));
        if (this.getClass() == other.getClass()) {
            double[] xd = array.clone();
            for (int i = 0; i < xd.length; i++) xd[i] += other.get(i, i);
            return new DiagonalDoubleMatrix(xd);
        }
        return FullDoubleMatrix.add(this, other);
    }
    public double get(int row, int column) {
        shape.assertInShape(row, column);
        return row == column ? array[row] : 0;
    }
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder("Type: diagonal\nShape: " +
                shape.rows + " x " + shape.columns + "\n");
        for (int i = 0, n = array.length; i < n; i++) {
            if (i == 0) {
                s.append(array[i]);
                if (n == 2) s.append(" 0\n");
                if (n == 3) s.append(" 0 0\n");
                if (n >  3) s.append(" 0 ... 0\n");
            } else if (i == n - 1) {
                if (n == 2) s.append("0 ");
                if (n == 3) s.append("0 0 ");
                if (n >  3) s.append("0 ... 0 ");
                s.append(array[i] + "\n");
            } else {
                if (i == 1) s.append("0 ");
                if (i == 2) s.append("0 0 ");
                if (i >  2) s.append("0 ... 0 ");
                s.append(array[i]);
                if (n - i - 1 == 1) s.append(" 0");
                if (n - i - 1 == 2) s.append(" 0 0");
                if (n - i - 1 >  2) s.append(" 0 ... 0");
                s.append("\n");
            }
        }
        return s.toString();
    }
}
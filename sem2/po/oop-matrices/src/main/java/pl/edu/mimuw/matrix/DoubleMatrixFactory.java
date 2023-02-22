package pl.edu.mimuw.matrix;

public class DoubleMatrixFactory {
    private DoubleMatrixFactory() {}
    public static IDoubleMatrix sparse(Shape shape, MatrixCellValue... values){
        return new SparseDoubleMatrix(shape, values);
    }
    public static IDoubleMatrix full(double[][] values) {
        return new FullDoubleMatrix(values);
    }
    public static IDoubleMatrix identity(int size) {
        return new IdentityDoubleMatrix(size);
    }
    public static IDoubleMatrix diagonal(double... diagonalValues) {
        return new DiagonalDoubleMatrix(diagonalValues);
    }
    public static IDoubleMatrix antiDiagonal(double... antiDiagonalValues) {
        return new AntiDiagonalDoubleMatrix(antiDiagonalValues);
    }
    public static IDoubleMatrix vector(double... values){
        double[][] xd = new double[values.length][1];
        for (int i = 0; i < values.length; i++) xd[i][0] = values[i];
        return new FullDoubleMatrix(xd);
    }
    public static IDoubleMatrix zero(Shape shape) {
        return new ConstDoubleMatrix(shape, 0);
    }
    public static IDoubleMatrix constant(Shape shape, double xd) {
        return new ConstDoubleMatrix(shape, xd);
    }
    public static IDoubleMatrix samerows(Shape shape, double ...xd) {
        return new SameRowsDoubleMatrix(shape, xd);
    }
}


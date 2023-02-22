package pl.edu.mimuw.matrix;

public interface IDoubleMatrix {
    IDoubleMatrix times(IDoubleMatrix other);
    IDoubleMatrix times(double scalar);
    IDoubleMatrix plus(IDoubleMatrix other);
    IDoubleMatrix plus(double scalar);
    IDoubleMatrix minus(IDoubleMatrix other);
    IDoubleMatrix minus(double scalar);
    double get(int row, int column);
    double[][] data();
    double normOne();
    double normInfinity();
    double frobeniusNorm();
    String toString();
    Shape shape();
}


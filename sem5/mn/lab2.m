function [x,l] = powermethod(A, l1, v1, tol, maxit)
  N = size(A, 1);
  x = rand(N, 1);
  A = sparse(A);
  Ax = A * x;
  for i = 1:maxit
    x = Ax - l1 * v1 * (v1' * x);
    x = x / norm(x);
    Ax = A * x;
    l = x' * Ax;
    if norm(Ax - l*x) < tol
      break;
    end
  end
end

function x = fullsolve(b, eps)
  n = length(b);
  u = ones(n, 1);
  vt = (u * eps)';

  Ta = u * (1 - eps);
  Tb = u * (6 - eps);

  x1 = TriDiagonal_Matrix_Algorithm(Ta, Tb, Ta, b);
  x2 = TriDiagonal_Matrix_Algorithm(Ta, Tb, Ta, u);
  x = x1 - x2 * (vt * x1) / (1 + vt * x2);
end

% https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
function x = TriDiagonal_Matrix_Algorithm(a, b, c, d)
  n = length(d);
  x = zeros(n, 1);
  for i=2:n
    w = a(i) / b(i-1);
    b(i) = b(i) - w * c(i-1);
    d(i) = d(i) - w * d(i-1);
  end
  x(n) = d(n) / b(n);
  for i=n-1:-1:1
    x(i) = (d(i) - c(i) * x(i+1)) / b(i);
  end
end

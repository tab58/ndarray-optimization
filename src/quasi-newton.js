'use strict';

var ndarray = require('ndarray');
var blas1 = require('ndarray-blas-level1');
var blas2 = require('ndarray-blas-level2');
var lineSearch = require('./line-search.js');
var gradientSelect = require('./lib/gradient-select.js');
var rankUpdater = require('./lib/update-select.js');

var MAX_ITERATIONS = 200;
module.exports = function quasiNewton (options) {
  if (!options.start) {
    throw new Error('Undefined start position.');
  }
  if (!options.objective.func) {
    throw new Error('Undefined objective function.');
  }
  var maxIterations = options.solution.maxIterations || MAX_ITERATIONS;
  var TOLERANCE = options.solution.tolerance || 1e-11;
  var evaluateDerivative = gradientSelect(options);
  var F = options.objective.func;
  var updateInverse = rankUpdater(options);

  var x0 = options.start;
  var n = x0.shape[0];
  var x1 = ndarray(new Float64Array(n), [n]);
  var dx = ndarray(new Float64Array(n), [n]);
  var grad0 = ndarray(new Float64Array(n), [n]);
  var grad1 = ndarray(new Float64Array(n), [n]);
  var y = ndarray(new Float64Array(n), [n]);
  var N = ndarray(new Float64Array(n * n), [n, n]);
  var f0 = Number.POSITIVE_INFINITY;
  var f1 = F(x0);
  for (var i = 0; i < n; ++i) {
    for (var j = 0; j < n; ++j) {
      N.set(i, j, i === j ? 1 : 0);
    }
  }
  var gradNorm = evaluateDerivative(x0, grad0);
  blas1.cpsc(-1.0 / gradNorm, grad0, y);

  var iter = 0;
  var temp1;
  var temp2;
  while (Math.abs(f1 - f0) > TOLERANCE && Math.abs(gradNorm) > TOLERANCE && iter++ <= maxIterations) {
    lineSearch.parabolicApprox(x0, y, F, x1);
    gradNorm = evaluateDerivative(x1, grad1);
    f0 = f1;
    f1 = F(x1);
    blas1.copy(x1, dx);
    blas1.axpy(-1.0, x0, dx);
    blas1.copy(grad1, y);
    blas1.axpy(-1.0, grad0, y);
    updateInverse(N, y, dx);
    blas2.gemv(-1.0 / gradNorm, N, grad1, 0.0, y);
    temp1 = grad0;
    grad0 = grad1;
    grad1 = temp1;
    temp2 = x0;
    x0 = x1;
    x1 = temp2;
  }

  var solutionValid = (Math.abs(f1 - f0) > TOLERANCE) || (Math.abs(gradNorm) > TOLERANCE);
  var results = {
    'solutionValid': solutionValid,
    'iterations': iter,
    'objective': f1,
    'gradNorm': gradNorm
  };
  return results;
};

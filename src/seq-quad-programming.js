'use strict';

// var ndarray = require('ndarray');
var defaults = require('./global-defaults.js');

module.exports = function seqQuadProgramming (options) {
  // property checks
  if (!options.objective) {
    throw new Error('Undefined optimization objective.');
  }
  if (!options.objective.start) {
    throw new Error('Undefined start position.');
  }
  if (!options.objective.func) {
    throw new Error('Undefined objective function.');
  }

  var maxIterations = defaults.MAX_ITERATIONS;
  var tolerance = defaults.tolerance;
  if (options.solution) {
    if (options.solution.maxIterations &&
      !isNaN(options.solution.maxIterations)) {
      maxIterations = options.solution.maxIterations;
    } else {
      console.warn('Maximum iterations capped at default of ' + maxIterations + '.');
    }
    if (options.solution.tolerance &&
      !isNaN(options.solution.tolerance)) {
      tolerance = options.solution.tolerance;
    } else {
      console.warn('Numerical tolerance is default of ' + tolerance + '.');
    }
  }

  // algorithm

  // default return false to signal incomplete implementation
  return false;
};

'use strict';

module.exports.unconstrained = {
  steepestDescent: require('./src/steepest-descent.js'),
  quasiNewton: require('./src/quasi-newton.js')
};

module.exports.constrained = {
  kuhnTucker: require('./src/kuhn-tucker.js')
};

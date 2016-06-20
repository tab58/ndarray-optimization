'use strict';

var chai = require('chai');
var ndarray = require('ndarray');
var kuhnTucker = require('../src/kuhn-tucker.js');
console.log(kuhnTucker);

// var TOLERANCE = 1e-11;
// var MAX_ITERATIONS = 5;

describe('Constrained - Kuhn-Tucker Conditions', function () {
  it('1 Constraint - Linear, Equality', function () {
    var F = function (X) {
      var x1 = X.get(0, 0);
      var x2 = X.get(1, 0);
      var f = 2 * x1 * x1 + 4 * x2 * x2;
      return f;
    };

    var G1 = function (X) {
      var x1 = X.get(0, 0);
      var x2 = X.get(1, 0);
      var f = 3 * x1 + 2 * x2 - 12;
      return f;
    };

    var options = {
      objective: {
        func: F,
        gradient: {
          func: 'centralDifference',
          delta: 0.01
        }
      },
      constraints: {
        equality: [ {
          func: G1,
          gradient: {
            func: 'centralDifference',
            delta: 0.01
          }
        }]
      },
      solution: {
        tolerance: 1e-10
      }
    };
    var result = kuhnTucker(options, ndarray(new Float64Array([0, 0]), [2, 1]));
    chai.assert(!result, 'Problem should not meet the Kuhn-Tucker conditions.');
    result = kuhnTucker(options, ndarray(new Float64Array([3.2727272727, 1.0909090909]), [2, 1]));
    chai.assert(result, 'Problem should meet the Kuhn-Tucker conditions.');
  });
  // it('1 Constraint - Linear, Inequality', function () {
  //   var F = function (X) {
  //     var x1 = X.get(0);
  //     var x2 = X.get(1);
  //     var f = x1 * x1 + x2 * x2;
  //     return f;
  //   };

  //   var dF = function (X, grad) {
  //     var x1 = X.get(0);
  //     var x2 = X.get(1);
  //     var g1 = 2 * x1;
  //     var g2 = 2 * x2;
  //     grad.set(0, g1);
  //     grad.set(1, g2);
  //     return Math.sqrt(g1 * g1 + g2 * g2);
  //   };

  //   var G1 = function (X) {
  //     var x1 = X.get(0);
  //     var x2 = X.get(1);
  //     var f = x1 + 4 * x2 - 12;
  //     return f;
  //   };

  //   var options = {
  //     objective: {
  //       func: F,
  //       gradient: {
  //         func: dF,
  //         delta: 0
  //       }
  //     },
  //     constraints: {
  //       inequality: [ {
  //         func: G1,
  //         gradient: {
  //           func: 'centralDifference',
  //           delta: 0.01
  //         }
  //       }]
  //     },
  //     solution: {
  //       tolerance: 1e-10
  //     }
  //   };
  //   var result = kuhnTucker(options, ndarray(new Float64Array([0.7059, 2.8235]), [2, 1]));
  //   chai.assert(result, 'Problem does not meet the Kuhn-Tucker conditions.');
  // });
});

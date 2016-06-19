'use strict';

var chai = require('chai');
var ndarray = require('ndarray');
var quasiNewton = require('../src/quasi-newton.js');

var TOLERANCE = 1e-11;
var MAX_ITERATIONS = 5;

describe('Constrained - Kuhn-Tucker Conditions', function () {
  it('1 Linear Constraint', function () {
    var F = function (X) {
      var x1 = X.get(0);
      var x2 = X.get(1);
      var f = 2 * x1 * x1 + 4 * x2 * x2;
      return f;
    };

    var dF = function (X, grad) {
      var x1 = X.get(0);
      var x2 = X.get(1);
      var g1 = 2 * x1;
      var g2 = 8 * x2;
      grad.set(0, g1);
      grad.set(1, g2);
      return Math.sqrt(g1 * g1 + g2 * g2);
    };

    var G1 = function (X) {
      var x1 = X.get(0);
      var x2 = X.get(1);
      var f = 3 * x1 + 2 * x2 - 12;
      return f;
    };

    options = {
      objective: {
        func: F,
        gradient: {
          func: dF,
          delta: 0
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

    
  });
});
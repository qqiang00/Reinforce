var R = {}; // the Recurrent library 网络相关库包括矩阵运算、随机数等

(function (global) {
  "use strict";

  // 工具函数，在条件不成立的情况下抛出消息提示异常
  function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  // 产生一系列的随机数
  var isVReturned = false;
  var v_val = 0.0;
  var gaussRandom = function () {
    if (isVReturned) {
      isVReturned = false;
      return v_val;
    }
    var u = 2 * Math.random() - 1;
    var v = 2 * Math.random() - 1;
    var r = u * u + v * v;
    if (r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2 * Math.log(r) / r);
    v_val = v * c; // cache this
    isVReturned = true;
    return u * c;
  }
  // 产生a,b之间的随机浮点数
  var random = function (a, b) { return Math.random() * (b - a) + a; }
  // 产生a,b之间的随机正数，包括a，不包括b?
  var randInt = function (a, b) { return Math.floor(Math.random() * (b - a) + a); }
  // 产生符合参数决定的高斯分布的随机浮点数
  var randNorm = function (mu, std) { return mu + gaussRandom() * std; }

  // helper function returns array of zeros of length n
  // and uses typed arrays if available
  // 产生长度为n的，元素值均为0的数组，该数组也可以用来存储矩阵。
  var zeros = function (n) {
    if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
    if (typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for (var i = 0; i < n; i++) { arr[i] = 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
  }

  var ones = function(n) {
    if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
    var arr = new Array(n);
    for (var i = 0; i < n; i++) { arr[i] = 1; }
    return arr;
  }

  // Mat holds a matrix 产生行列分别为n,d元素均为0矩阵
  var Mat = function (n, d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);   // 矩阵包括一个临时矩阵存储dw
  }
  Mat.prototype = {
    get: function (row, col) {
      // slow but careful accessor function
      // we want row-major order
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      return this.w[ix];
    },
    set: function (row, col, v) {
      // slow but careful accessor function
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      this.w[ix] = v;
    },
    // 对w内所有元素求和
    sum: function () {
      var v = 0.0;
      for (var q = 0, n = this.w.length; q < n; q++) {
        v += this.w[q];
      }
      return v;
    },
    // 从数组arr拷贝至当前矩阵
    setFrom: function (arr) {
      for (var i = 0, n = arr.length; i < n; i++) {
        this.w[i] = arr[i];
      }
    },

    // 将矩阵m第i列的数据赋值给当前矩阵的第i列
    setColumn: function (m, i) {
      for (var q = 0, n = m.w.length; q < n; q++) {
        this.w[(this.d * q) + i] = m.w[q];
      }
    },

    toJSON: function () {
      var json = {};
      json['n'] = this.n;
      json['d'] = this.d;
      json['w'] = this.w;
      return json;
    },

    fromJSON: function (json) {
      this.n = json.n;
      this.d = json.d;
      this.w = zeros(this.n * this.d);
      this.dw = zeros(this.n * this.d);
      for (var i = 0, n = this.n * this.d; i < n; i++) {
        this.w[i] = json.w[i]; // copy over weights
      }
    }
  }

  // 拷贝矩阵
  var copyMat = function (b) {
    var a = new Mat(b.n, b.d);
    a.setFrom(b.w);
    return a;
  }

  // 拷贝整个网络
  var copyNet = function (net) {
    // nets are (k,v) pairs with k = string key, v = Mat()
    var new_net = {};
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        new_net[p] = copyMat(net[p]);
      }
    }
    return new_net;
  }

  // 更新矩阵m，用dw更新w，以一定的学习率alpha
  // 这里应用的是w减去dw更新w
  var updateMat = function (m, alpha) {
    // updates in place
    for (var i = 0, n = m.n * m.d; i < n; i++) {
      if (m.dw[i] !== 0) {
        m.w[i] += - alpha * m.dw[i];
        m.dw[i] = 0;
      }
    }
  }

  // 以一定的学习率对整个网络的矩阵进行更新
  var updateNet = function (net, alpha) {
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        updateMat(net[p], alpha);
      }
    }
  }

  var netToJSON = function (net) {
    var j = {};
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        j[p] = net[p].toJSON();
      }
    }
    return j;
  }

  var netFromJSON = function (j) {
    var net = {};
    for (var p in j) {
      if (j.hasOwnProperty(p)) {
        net[p] = new Mat(1, 1); // not proud of this
        net[p].fromJSON(j[p]);
      }
    }
    return net;
  }

  // 将整个网络的梯度矩阵dw所有元素设为0
  var netZeroGrads = function (net) {
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        var mat = net[p];
        gradFillConst(mat, 0);
      }
    }
  }

  // 把整个网络的dw矩阵转换成一个列向量？
  var netFlattenGrads = function (net) {
    var n = 0;
    for (var p in net) { if (net.hasOwnProperty(p)) { var mat = net[p]; n += mat.dw.length; } }
    var g = new Mat(n, 1);
    var ix = 0;
    for (var p in net) {
      if (net.hasOwnProperty(p)) {
        var mat = net[p];
        for (var i = 0, m = mat.dw.length; i < m; i++) {
          g.w[ix] = mat.dw[i];
          ix++;
        }
      }
    }
    return g;
  }

  // return Mat but filled with random numbers from gaussian
  // 建立一个矩阵，并以高斯分布填充
  var RandMat = function (n, d, mu, std) {
    var m = new Mat(n, d);
    fillRandn(m, mu, std);
    //fillRand(m,-std,std); // kind of :P
    return m;
  }

  var OneMat = function (n, d) {
    var m = new Mat(n, d);
    fillOne(m);
    return m;
  }
  // Mat utils
  // fill matrix with random gaussian numbers
  var fillRandn = function (m, mu, std) { for (var i = 0, n = m.w.length; i < n; i++) { m.w[i] = randNorm(mu, std); } }
  var fillRand = function (m, lo, hi) { for (var i = 0, n = m.w.length; i < n; i++) { m.w[i] = random(lo, hi); } }
  var gradFillConst = function (m, c) { for (var i = 0, n = m.dw.length; i < n; i++) { m.dw[i] = c } }
  var fillOne = function(m) { for (var i = 0, n = m.w.length; i < n; i++ ) { m.w[i] = 1.0; }}

  // 图主要用来执行矩阵运算，并记住运算形式给出反向传播算法
  // Transformer definitions
  var Graph = function (needs_backprop) {
    // 默认需要反向
    if (typeof needs_backprop === 'undefined') { needs_backprop = true; }
    this.needs_backprop = needs_backprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    // 存储一个方法列表，在反向传播的时候按照前向次序依次执行
    this.backprop = [];
    this.lastM = undefined;
  }

  Graph.prototype = {
    backward: function () {
      for (var i = this.backprop.length - 1; i >= 0; i--) {
        this.backprop[i](); // tick!
      }
    },

    // 从矩阵m中抽第ix行数据返回作为列向量
    rowPluck: function (m, ix) {
      // pluck a row of m with index ix and return it as col vector
      assert(ix >= 0 && ix < m.n);
      var d = m.d;  // 矩阵m的列数
      var out = new Mat(d, 1) //行数为原矩阵列数的列向量;
      for (var i = 0, n = d; i < n; i++) { out.w[i] = m.w[d * ix + i]; } // copy over the data

      // 该操作的梯度
      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0, n = d; i < n; i++) { m.dw[d * ix + i] += out.dw[i]; }
        }
        this.backprop.push(backward);
      }
      return out;
    },

    // 对矩阵内元素进行tanh运算
    tanh: function (m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for (var i = 0; i < n; i++) {
        out.w[i] = Math.tanh(m.w[i]);
      }

      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0; i < n; i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },

    // sigmoid函数及梯度函数
    sigmoid: function (m) {
      // sigmoid nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for (var i = 0; i < n; i++) {
        out.w[i] = sig(m.w[i]);
      }

      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0; i < n; i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },

    // relu函数
    relu: function (m) {
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for (var i = 0; i < n; i++) {
        out.w[i] = Math.max(0, m.w[i]); // relu
      }
      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0; i < n; i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },

    // 矩阵相乘
    mul: function (m1, m2) {
      // multiply matrices m1 * m2
      assert(m1.d === m2.n, 'matmul dimensions misaligned');

      var n = m1.n;
      var d = m2.d;
      var out = new Mat(n, d);
      for (var i = 0; i < m1.n; i++) { // loop over rows of m1
        for (var j = 0; j < m2.d; j++) { // loop over cols of m2
          var dot = 0.0;
          for (var k = 0; k < m1.d; k++) { // dot product loop
            dot += m1.w[m1.d * i + k] * m2.w[m2.d * k + j];
          }
          out.w[d * i + j] = dot;
        }
      }

      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0; i < m1.n; i++) { // loop over rows of m1
            for (var j = 0; j < m2.d; j++) { // loop over cols of m2
              for (var k = 0; k < m1.d; k++) { // dot product loop
                var b = out.dw[d * i + j];
                m1.dw[m1.d * i + k] += m2.w[m2.d * k + j] * b;
                m2.dw[m2.d * k + j] += m1.w[m1.d * i + k] * b;
              }
            }
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },

    // 矩阵相加
    add: function (m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for (var i = 0, n = m1.w.length; i < n; i++) {
        out.w[i] = m1.w[i] + m2.w[i];
      }
      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0, n = m1.w.length; i < n; i++) {
            m1.dw[i] += out.dw[i];
            m2.dw[i] += out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },

    // 矩阵点积：对应元素相乘求和
    dot: function (m1, m2) {
      // m1 m2 are both column vectors
      assert(m1.w.length === m2.w.length);
      var out = new Mat(1, 1);
      var dot = 0.0;
      for (var i = 0, n = m1.w.length; i < n; i++) {
        dot += m1.w[i] * m2.w[i];
      }
      out.w[0] = dot;
      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0, n = m1.w.length; i < n; i++) {
            m1.dw[i] += m2.w[i] * out.dw[0];
            m2.dw[i] += m1.w[i] * out.dw[0];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },

    // 矩阵内对应元素相乘
    eltmul: function (m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for (var i = 0, n = m1.w.length; i < n; i++) {
        out.w[i] = m1.w[i] * m2.w[i];
      }
      if (this.needs_backprop) {
        var backward = function () {
          for (var i = 0, n = m1.w.length; i < n; i++) {
            m1.dw[i] += m2.w[i] * out.dw[i];
            m2.dw[i] += m1.w[i] * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
  }

  var softmax = function (m) {
    var out = new Mat(m.n, m.d); // probability volume
    var maxval = -999999;
    for (var i = 0, n = m.w.length; i < n; i++) { if (m.w[i] > maxval) maxval = m.w[i]; }

    var s = 0.0;
    for (var i = 0, n = m.w.length; i < n; i++) {
      out.w[i] = Math.exp(m.w[i] - maxval);
      s += out.w[i];
    }
    for (var i = 0, n = m.w.length; i < n; i++) { out.w[i] /= s; }

    // no backward pass here needed
    // since we will use the computed probabilities outside
    // to set gradients directly on m
    return out;
  }

  var Solver = function () {
    this.decay_rate = 0.999;
    this.smooth_eps = 1e-8;
    this.step_cache = {};
  }
  Solver.prototype = {
    step: function (model, step_size, regc, clipval) {
      // perform parameter update
      var solver_stats = {};
      var num_clipped = 0;
      var num_tot = 0;
      for (var k in model) {
        if (model.hasOwnProperty(k)) {
          var m = model[k]; // mat ref
          if (!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
          var s = this.step_cache[k];
          for (var i = 0, n = m.w.length; i < n; i++) {

            // rmsprop adaptive learning rate
            var mdwi = m.dw[i];
            s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

            // gradient clip
            if (mdwi > clipval) {
              mdwi = clipval;
              num_clipped++;
            }
            if (mdwi < -clipval) {
              mdwi = -clipval;
              num_clipped++;
            }
            num_tot++;

            // update (and regularize)
            m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
            m.dw[i] = 0; // reset gradients for next iteration
          }
        }
      }
      solver_stats['ratio_clipped'] = num_clipped * 1.0 / num_tot;
      return solver_stats;
    }
  }

  var initLSTM = function (input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    for (var d = 0; d < hidden_sizes.length; d++) { // loop over depths
      var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      var hidden_size = hidden_sizes[d];

      // gates parameters
      model['Wix' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model['Wih' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model['bi' + d] = new Mat(hidden_size, 1);
      model['Wfx' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model['Wfh' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model['bf' + d] = new Mat(hidden_size, 1);
      model['Wox' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model['Woh' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model['bo' + d] = new Mat(hidden_size, 1);
      // cell write params
      model['Wcx' + d] = new RandMat(hidden_size, prev_size, 0, 0.08);
      model['Wch' + d] = new RandMat(hidden_size, hidden_size, 0, 0.08);
      model['bc' + d] = new Mat(hidden_size, 1);
    }
    // decoder params
    model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
    model['bd'] = new Mat(output_size, 1);
    return model;
  }

  var forwardLSTM = function (G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    if (prev == null || typeof prev.h === 'undefined') {
      var hidden_prevs = [];
      var cell_prevs = [];
      for (var d = 0; d < hidden_sizes.length; d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d], 1));
        cell_prevs.push(new R.Mat(hidden_sizes[d], 1));
      }
    } else {
      var hidden_prevs = prev.h;
      var cell_prevs = prev.c;
    }

    var hidden = [];
    var cell = [];
    for (var d = 0; d < hidden_sizes.length; d++) {

      var input_vector = d === 0 ? x : hidden[d - 1];
      var hidden_prev = hidden_prevs[d];
      var cell_prev = cell_prevs[d];

      // input gate
      var h0 = G.mul(model['Wix' + d], input_vector);
      var h1 = G.mul(model['Wih' + d], hidden_prev);
      var input_gate = G.sigmoid(G.add(G.add(h0, h1), model['bi' + d]));

      // forget gate
      var h2 = G.mul(model['Wfx' + d], input_vector);
      var h3 = G.mul(model['Wfh' + d], hidden_prev);
      var forget_gate = G.sigmoid(G.add(G.add(h2, h3), model['bf' + d]));

      // output gate
      var h4 = G.mul(model['Wox' + d], input_vector);
      var h5 = G.mul(model['Woh' + d], hidden_prev);
      var output_gate = G.sigmoid(G.add(G.add(h4, h5), model['bo' + d]));

      // write operation on cells
      var h6 = G.mul(model['Wcx' + d], input_vector);
      var h7 = G.mul(model['Wch' + d], hidden_prev);
      var cell_write = G.tanh(G.add(G.add(h6, h7), model['bc' + d]));

      // compute new cell activation
      var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
      var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
      var cell_d = G.add(retain_cell, write_cell); // new cell contents

      // compute hidden state as gated, saturated cell activations
      var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

      hidden.push(hidden_d);
      cell.push(cell_d);
    }

    // one decoder to outputs at end
    var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]), model['bd']);

    // return cell memory, hidden representation and output
    return { 'h': hidden, 'c': cell, 'o': output };
  }
  ////////////////////////////////////////
  var sig = function (x) {
    // helper function for computing sigmoid
    return 1.0 / (1 + Math.exp(-x));
  }


  var maxi = function (w) {
    // 返回一个数组最大值的索引
    var maxv = w[0];
    var maxix = 0;
    for (var i = 1, n = w.length; i < n; i++) {
      var v = w[i];
      if (v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  }

  // 从数组w按照w存储的概率分布取样
  var samplei = function (w) {
    // sample argmax from w, assuming w are 
    // probabilities that sum to one
    var r = random(0, 1);
    var x = 0.0;
    var i = 0;
    while (true) {
      x += w[i];
      if (x > r) { return i; }
      i++;
    }
    //return w.length - 1; // pretty sure we should never get here?
  }

  // various utils
  global.assert = assert;
  global.zeros = zeros;
  global.ones = ones;
  global.maxi = maxi;
  global.samplei = samplei;
  global.randInt = randInt;
  global.random = random;
  global.randNorm = randNorm;
  global.sig = sig;
  global.softmax = softmax;
  // classes
  global.Mat = Mat;
  global.RandMat = RandMat;
  global.OneMat = OneMat;
  global.forwardLSTM = forwardLSTM;
  global.initLSTM = initLSTM;
  // more utils
  global.updateMat = updateMat;
  global.updateNet = updateNet;
  global.copyMat = copyMat;
  global.copyNet = copyNet;
  global.netToJSON = netToJSON;
  global.netFromJSON = netFromJSON;
  global.netZeroGrads = netZeroGrads;
  global.netFlattenGrads = netFlattenGrads;
  // optimization
  global.Solver = Solver;
  global.Graph = Graph;
})(R);

// END OF RECURRENTJS

var RL = {};
(function (global) {
  "use strict";

  // syntactic sugar function for getting default parameter values
  var getopt = function (opt, field_name, default_value) {
    if (typeof opt === 'undefined') { return default_value; }
    return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
  }

  // 从前面一大块的R对象里借用一些常用的函数
  var zeros = R.zeros; // inherit these
  var assert = R.assert;
  var randInt = R.randInt;
  var random = R.random;

  // 将一个数组arr中的每一个元素设定为常数c
  var setConst = function (arr, c) {
    for (var i = 0, n = arr.length; i < n; i++) {
      arr[i] = c;
    }
  }

  // 参数p应该是一个概率分布，函数返回一个随机索引
  var sampleWeighted = function (p) {
    var r = Math.random();  // 得到0-1之间的数
    var c = 0.0;
    for (var i = 0, n = p.length; i < n; i++) {
      c += p[i];
      if (c >= r) { return i; }
    }
    assert(false, 'wtf');
  }

  // ------
  // AGENTS
  // ------

  // DPAgent performs Value Iteration
  // - can also be used for Policy Iteration if you really wanted to
  // - requires model of the environment :(
  // - does not learn from experience :(
  // - assumes finite MDP :(
  // DPAgent接受环境对象和参设设置对象作为参数
  var DPAgent = function (env, opt) {
    this.V = null; // state value function
    this.P = null; // policy distribution \pi(s,a)
    this.env = env; // store pointer to environment
    // 从opt对象获取'gamma'参数，找不到的话用默认值0.75代替
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.reset();
  }
  DPAgent.prototype = {
    reset: function () {
      // reset the agent's policy and value function
      this.nfs = this.env.getNumStates();
      this.nfa = this.env.getMaxNumActions();
      this.V = zeros(this.nfs);
      this.P = zeros(this.nfs * this.nfa);
      // initialize uniform random policy
      for (var s = 0; s < this.nfs; s++) {
        // 在环境模型里过滤了不能操作的行为
        var poss = this.env.allowedActions(s);
        //var poss = this.env.allActions(s);
        for (var i = 0, n = poss.length; i < n; i++) {
          this.P[poss[i] * this.nfs + s] = 1.0 / poss.length;
          //this.P[poss[5]*this.nfs+s]=1.0;
        }
      }
    },
    act: function (s) {
      // behave according to the learned policy
      var poss = this.env.allowedActions(s);
      //var poss = this.env.allActions(s);
      var ps = [];
      for (var i = 0, n = poss.length; i < n; i++) {
        var a = poss[i];
        var prob = this.P[a * this.nfs + s];
        ps.push(prob);
      }
      var maxi = sampleWeighted(ps);
      return poss[maxi];
    },
    learn: function () {
      // perform a single round of value iteration
      self.evaluatePolicy(); // writes this.V
      self.updatePolicy(); // writes this.P
    },
    evaluatePolicy: function () {
      // perform a synchronous update of the value function
      var Vnew = zeros(this.nfs);
      for (var s = 0; s < this.nfs; s++) {
        // integrate over actions in a stochastic policy
        // note that we assume that policy probability mass over allowed actions sums to one
        var v = 0.0;
        var poss = this.env.allowedActions(s);
        //var poss = this.env.allActions(s);
        for (var i = 0, n = poss.length; i < n; i++) {
          var a = poss[i];
          var prob = this.P[a * this.nfs + s]; // probability of taking action under policy
          if (prob === 0) { continue; } // no contribution, skip for speed
          var ns = this.env.nextStateDistribution(s, a);
          var rs = this.env.reward(s, a, ns); // reward for s->a->ns transition
          v += prob * (rs + this.gamma * this.V[ns]);
        }
        Vnew[s] = v;

      }
      this.V = Vnew; // swap
    },
    updatePolicy: function () {
      // update policy to be greedy w.r.t. learned Value function
      for (var s = 0; s < this.nfs; s++) {
        //var poss = this.env.allActions(s);
        var poss = this.env.allowedActions(s);
        // compute value of taking each allowed action
        var vmax, nmax;
        var vs = [];
        for (var i = 0, n = poss.length; i < n; i++) {
          var a = poss[i];
          var ns = this.env.nextStateDistribution(s, a);
          var rs = this.env.reward(s, a, ns);
          var v = rs + this.gamma * this.V[ns];
          vs.push(v);
          if (i === 0 || v > vmax) { vmax = v; nmax = 1; }
          else if (v === vmax) { nmax += 1; }
        }
        // update policy smoothly across all argmaxy actions
        for (var i = 0, n = poss.length; i < n; i++) {
          var a = poss[i];
          this.P[a * this.nfs + s] = (vs[i] === vmax) ? 1.0 / nmax : 0.0;
        }
      }
    },
  }

  // QAgent uses TD (Q-Learning, SARSA)
  // - does not require environment model :)
  // - learns from experience :)
  var TDAgent = function (env, opt) {
    this.update = getopt(opt, 'update', 'qlearn'); // qlearn | sarsa
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

    // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
    this.smooth_policy_update = getopt(opt, 'smooth_policy_update', false);
    this.beta = getopt(opt, 'beta', 0.01); // learning rate for policy, if smooth updates are on

    // eligibility traces
    this.lambda = getopt(opt, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
    this.replacing_traces = getopt(opt, 'replacing_traces', true);

    // optional optimistic initial values
    this.q_init_val = getopt(opt, 'q_init_val', 0);

    this.planN = getopt(opt, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)

    this.Q = null; // state action value function
    this.P = null; // policy distribution \pi(s,a)
    this.e = null; // eligibility trace
    this.env_model_s = null;; // environment model (s,a) -> (s',r)
    this.env_model_r = null;; // environment model (s,a) -> (s',r)
    this.env = env; // store pointer to environment
    this.reset();
  }

  TDAgent.prototype = {
    reset: function () {
      // reset the agent's policy and value function
      this.nfs = this.env.getNumStates();
      this.nfa = this.env.getMaxNumActions();
      this.Q = zeros(this.nfs * this.nfa);
      if (this.q_init_val !== 0) { setConst(this.Q, this.q_init_val); }
      this.P = zeros(this.nfs * this.nfa);
      this.e = zeros(this.nfs * this.nfa);

      // model/planning vars
      this.env_model_s = zeros(this.nfs * this.nfa);
      setConst(this.env_model_s, -1); // init to -1 so we can test if we saw the state before
      this.env_model_r = zeros(this.nfs * this.nfa);
      this.sa_seen = [];
      this.pq = zeros(this.nfs * this.nfa);

      // initialize uniform random policy
      for (var s = 0; s < this.nfs; s++) {
        var poss = this.env.allowedActions(s);
        for (var i = 0, n = poss.length; i < n; i++) {
          this.P[poss[i] * this.nfs + s] = 1.0 / poss.length;
        }
      }
      // agent memory, needed for streaming updates
      // (s0,a0,r0,s1,a1,r1,...)
      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
    },
    resetEpisode: function () {
      // an episode finished
    },
    act: function (s) {
      // act according to epsilon greedy policy
      var poss = this.env.allowedActions(s);
      var probs = [];
      for (var i = 0, n = poss.length; i < n; i++) {
        probs.push(this.P[poss[i] * this.nfs + s]);
      }
      // epsilon greedy policy
      if (Math.random() < this.epsilon) {
        var a = poss[randInt(0, poss.length)]; // random available action
        this.explored = true;
      } else {
        var a = poss[sampleWeighted(probs)];
        this.explored = false;
      }
      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;
      return a;
    },
    learn: function (r1) {
      // takes reward for previous action, which came from a call to act()
      if (!(this.r0 == null)) {
        this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
        if (this.planN > 0) {
          this.updateModel(this.s0, this.a0, this.r0, this.s1);
          this.plan();
        }
      }
      this.r0 = r1; // store this for next update
    },
    updateModel: function (s0, a0, r0, s1) {
      // transition (s0,a0) -> (r0,s1) was observed. Update environment model
      var sa = a0 * this.nfs + s0;
      if (this.env_model_s[sa] === -1) {
        // first time we see this state action
        this.sa_seen.push(a0 * this.nfs + s0); // add as seen state
      }
      this.env_model_s[sa] = s1;
      this.env_model_r[sa] = r0;
    },
    plan: function () {

      // order the states based on current priority queue information
      var spq = [];
      for (var i = 0, n = this.sa_seen.length; i < n; i++) {
        var sa = this.sa_seen[i];
        var sap = this.pq[sa];
        if (sap > 1e-5) { // gain a bit of efficiency
          spq.push({ sa: sa, p: sap });
        }
      }
      spq.sort(function (a, b) { return a.p < b.p ? 1 : -1 });

      // perform the updates
      var nsteps = Math.min(this.planN, spq.length);
      for (var k = 0; k < nsteps; k++) {
        // random exploration
        //var i = randInt(0, this.sa_seen.length); // pick random prev seen state action
        //var s0a0 = this.sa_seen[i];
        var s0a0 = spq[k].sa;
        this.pq[s0a0] = 0; // erase priority, since we're backing up this state
        var s0 = s0a0 % this.nfs;
        var a0 = Math.floor(s0a0 / this.nfs);
        var r0 = this.env_model_r[s0a0];
        var s1 = this.env_model_s[s0a0];
        var a1 = -1; // not used for Q learning
        if (this.update === 'sarsa') {
          // generate random action?...
          var poss = this.env.allowedActions(s1);
          var a1 = poss[randInt(0, poss.length)];
        }
        this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
      }
    },
    learnFromTuple: function (s0, a0, r0, s1, a1, lambda) {
      var sa = a0 * this.nfs + s0;

      // calculate the target for Q(s,a)
      if (this.update === 'qlearn') {
        // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
        var poss = this.env.allowedActions(s1);
        var qmax = 0;
        for (var i = 0, n = poss.length; i < n; i++) {
          var s1a = poss[i] * this.nfs + s1;
          var qval = this.Q[s1a];
          if (i === 0 || qval > qmax) { qmax = qval; }
        }
        var target = r0 + this.gamma * qmax;
      } else if (this.update === 'sarsa') {
        // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
        var s1a1 = a1 * this.nfs + s1;
        var target = r0 + this.gamma * this.Q[s1a1];
      }

      if (lambda > 0) {
        // perform an eligibility trace update
        if (this.replacing_traces) {
          this.e[sa] = 1;
        } else {
          this.e[sa] += 1;
        }
        var edecay = lambda * this.gamma;
        var state_update = zeros(this.nfs);
        for (var s = 0; s < this.nfs; s++) {
          var poss = this.env.allowedActions(s);
          for (var i = 0; i < poss.length; i++) {
            var a = poss[i];
            var saloop = a * this.nfs + s;
            var esa = this.e[saloop];
            var update = this.alpha * esa * (target - this.Q[saloop]);
            this.Q[saloop] += update;
            this.updatePriority(s, a, update);
            this.e[saloop] *= edecay;
            var u = Math.abs(update);
            if (u > state_update[s]) { state_update[s] = u; }
          }
        }
        for (var s = 0; s < this.nfs; s++) {
          if (state_update[s] > 1e-5) { // save efficiency here
            this.updatePolicy(s);
          }
        }
        if (this.explored && this.update === 'qlearn') {
          // have to wipe the trace since q learning is off-policy :(
          this.e = zeros(this.nfs * this.nfa);
        }
      } else {
        // simpler and faster update without eligibility trace
        // update Q[sa] towards it with some step size
        var update = this.alpha * (target - this.Q[sa]);
        this.Q[sa] += update;
        this.updatePriority(s0, a0, update);
        // update the policy to reflect the change (if appropriate)
        this.updatePolicy(s0);
      }
    },
    updatePriority: function (s, a, u) {
      // used in planning. Invoked when Q[sa] += update
      // we should find all states that lead to (s,a) and upgrade their priority
      // of being update in the next planning step
      u = Math.abs(u);
      if (u < 1e-5) { return; } // for efficiency skip small updates
      if (this.planN === 0) { return; } // there is no planning to be done, skip.
      for (var si = 0; si < this.nfs; si++) {
        // note we are also iterating over impossible actions at all states,
        // but this should be okay because their env_model_s should simply be -1
        // as initialized, so they will never be predicted to point to any state
        // because they will never be observed, and hence never be added to the model
        for (var ai = 0; ai < this.nfa; ai++) {
          var siai = ai * this.nfs + si;
          if (this.env_model_s[siai] === s) {
            // this state leads to s, add it to priority queue
            this.pq[siai] += u;
          }
        }
      }
    },
    updatePolicy: function (s) {
      var poss = this.env.allowedActions(s);
      // set policy at s to be the action that achieves max_a Q(s,a)
      // first find the maxy Q values
      var qmax, nmax;
      var qs = [];
      for (var i = 0, n = poss.length; i < n; i++) {
        var a = poss[i];
        var qval = this.Q[a * this.nfs + s];
        qs.push(qval);
        if (i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
        else if (qval === qmax) { nmax += 1; }
      }
      // now update the policy smoothly towards the argmaxy actions
      var psum = 0.0;
      for (var i = 0, n = poss.length; i < n; i++) {
        var a = poss[i];
        var target = (qs[i] === qmax) ? 1.0 / nmax : 0.0;
        var ix = a * this.nfs + s;
        if (this.smooth_policy_update) {
          // slightly hacky :p
          this.P[ix] += this.beta * (target - this.P[ix]);
          psum += this.P[ix];
        } else {
          // set hard target
          this.P[ix] = target;
        }
      }
      if (this.smooth_policy_update) {
        // renomalize P if we're using smooth policy updates
        for (var i = 0, n = poss.length; i < n; i++) {
          var a = poss[i];
          this.P[a * this.nfs + s] /= psum;
        }
      }
    }
  }
  ///////////////////////////////////////////////////////////////////
  // a recommended agent definition:
  var Agent = function (env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.75);
    this.alpha = getopt(opt, 'epsilon', 0.1);
    this.env.env;

    this.state = undefined;
    this.action = undefined;
    this.reward1 = 0;
    // TODO: add your additional variables here
    this.reset();
  }
  Agent.prototype = {
    reset: function () {
      // get state and action feature number
      this.nfs = env.getNumOfStateFeatures();
      this.nfa = env.getMaxNumOfActionFeatures();
      // you can also get the detail features from env.
      // TODO:add your initialization code
    },
    observe: function () {
      // env should have a function with agent's next state returned
      return this.env.releasedObs();
    },
    performPolicy: function (state) {
      // generate an action for agent to execute or observe Q value
      // according to current state.
      var action = undefined;
      // TODO: add your code here.
      return action
    },
    performAction: function (action) {
      // execute an action and immediately get a reward from env
      // env should have a method defining its dynamics(model)
      // environment's dynamics should return a reward.
      this.action = action;
      this.reward1 = this.env.dynamics(this.action);
    },
    learn: function () {
      // agent lear the model by this.reward1.
    }
    // your addtional method here
  }
  // a recommened environment definition:
  var Env = function () {
    this.stateFeatures = ["feature1", "feature2"];//,"feature3"];
    this.actionFeatures = ["feature1", "feature2"];// ...
    this.reset();
  }
  Env.prototype = {
    reset: function () {
      // TODO: your environment initialization code here
      // TODO: add your variables that define environment.
    },
    getNumOfStateFeatures: function () { // 状态特征数
      return this.stateFeatures.length; // 
    },
    getMaxNumOfActionFeatures: function () { // 行为特征数
      return this.actionFeatures.length;
    },
    getState: function () {
      // return the environment state for visualization
      var s = [];
      // TODO: complete your elements in s
      return s
    },
    releaseObs: function () {
      var agentObs = [];
      // TODO: let agent see some elements of states.
      // or uncomment the following if you just want to return env's state
      // return this.getState();
      return agentObs;
    },
    dynamics: function (a) {
      // TODO: give your dynamics of environment here and return
      // reward to agnent.
      var reward = 0;
      // TWO works should be done in this function
      // 1: the update of environment states(including agent)
      // 2: give the reward to agent
      return reward;
    }
    // your can add extra functions here.
  }
  ///////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////
  var DQNAgent = function (env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

    this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
    this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
    this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);
    this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0);

    this.num_hidden_units = getopt(opt, 'num_hidden_units', 100);

    this.env = env;
    this.reset();


  }

  DQNAgent.prototype = {
    reset: function () {
      this.nh = this.num_hidden_units; // number of hidden units
      this.nfs = this.env.getNumOfStateFeatures();
      this.nfa = this.env.getMaxNumOfActionFeatures();

      // nets are hardcoded for now as key (str) -> Mat
      // not proud of this. better solution is to have a whole Net object
      // on top of Mats, but for now sticking with this
      this.net = {}; // net只存储了数据，具体数据运算交给Graph, 用来学习参数的。
      this.net.W1 = new R.RandMat(this.nh, this.nfs, 0, 0.01);
      this.net.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.net.W2 = new R.RandMat(this.nfa, this.nh, 0, 0.01);
      this.net.b2 = new R.Mat(this.nfa, 1, 0, 0.01);

      this.net2 = {}; // fixed Q，固定一段时间参数
      this.net2.W1 = new R.RandMat(this.nh, this.nfs, 0, 0.01);
      this.net2.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.net2.W2 = new R.RandMat(this.nfa, this.nh, 0, 0.01);
      this.net2.b2 = new R.Mat(this.nfa, 1, 0, 0.01);
      

      this.exp = []; // experience
      this.expi = 0; // where to insert

      this.t = 0;

      this.state = this.observe();
      this.action = null;
      this.reward1 = 0;

      //this.s1 = null; // st+1
      //this.a0 = null; // at
      //this.a1 = null; // at+1

      this.tderror = 0; // for visualization only...
    },
    observe: function () {
      return this.env.releasedObs();
    },
    performPolicy: function (slist) {
      // 建立一个列向量，行数为状态数
      var s = this.toMatS(slist);
      var a = undefined;
      // epsilon greedy policy 整体上是epsilon贪婪搜索
      if (Math.random() < this.epsilon) {
        a = randInt(0, this.nfa);
      } else {
        // greedy wrt Q function 根据Q结果贪婪选择行为
        var amat = this.forwardQ(this.net, s, false); // 对网络进行依次正向运算
        a = R.maxi(amat.w); // returns index of argmax action 
        // 获得网络输出向量最大值的索引进而货的行为
      }
      return a;
    },

    performAction: function (a) {
      this.action = a;
      this.reward1 = this.env.dynamics(this.action);
    },

    toJSON: function () {
      // save function
      var j = {};
      j.nh = this.nh;
      j.nfs = this.nfs;
      j.nfa = this.nfa;
      j.net = R.netToJSON(this.net);
      return j;
    },
    fromJSON: function (j) {
      // load function
      this.nh = j.nh;
      this.nfs = j.nfs;
      this.nfa = j.nfa;
      this.net = R.netFromJSON(j.net);
    },

    // 把状态送入网络得到网络输出
    forwardQ: function (net, s, needs_backprop) {
      // 从这里可以看得出网络的结构为：
      // 三层，第一到第二层用的是tanh函数，
      // 第二到第三层用的是简单的线性变换
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      this.lastG = G; // back this up. Kind of hacky isn't it
      return a2mat;
    },
    // 辅助函数，将状态列表转化为forwardQ方法里要求的Mat
    toMatS: function (s) {
      var m_s = new R.Mat(this.nfs, 1);
      m_s.setFrom(s);
      return m_s;
    },

    learn: function () {

      // perform an update on Q function
      if (this.alpha > 0) {

        // learn from this tuple to get a sense of how "surprising" it is to the agent
        var s1 = this.observe(); //得到最新的状态，但不更新到自己的状态中
        var tderror = this.learnFromTuple(this.state, this.action, this.reward1, s1);
        this.tderror = tderror; // a measure of surprise

        // decide if we should keep this experience in the replay
        if (this.t % this.experience_add_every === 0) {
          this.exp[this.expi] = [this.state, this.action, this.reward1, s1];
          this.expi += 1;
          if (this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
        }
        this.t += 1;

        // sample some additional experience from replay memory and learn from it
        for (var k = 0; k < this.learning_steps_per_iteration; k++) {
          var ri = randInt(0, this.exp.length); // todo: priority sweeps?
          var e = this.exp[ri];
          this.learnFromTuple(e[0], e[1], e[2], e[3])
        }
        // 经过一段时间随机学习后，把固定Q的网络更新为一直在更新参数的学习网络
        this.net2 = R.copyNet(this.net);
      }
    },

    learnFromTuple: function (s0, a0, r1, s1) {
      // want: Q(s,a) = r + gamma * max_a' Q(s',a')
      // 这里a1不需要使用
      // compute the target Q value // 得到一个行为空间向量
      var m_s1 = new R.Mat(this.nfs, 1);
      m_s1.setFrom(s1);
      var tmat = this.forwardQ(this.net2, m_s1, false);
      var qmax = r1 + this.gamma * tmat.w[R.maxi(tmat.w)];

      // now predict
      var m_s0 = new R.Mat(this.nfs, 1);
      m_s0.setFrom(s0);
      var pred = this.forwardQ(this.net, m_s0, true);

      var tderror = pred.w[a0] - qmax;

      // 限制了tderror范围
      var clamp = this.tderror_clamp;
      if (Math.abs(tderror) > clamp) {  // huber loss to robustify
        if (tderror > clamp) tderror = clamp;
        if (tderror < -clamp) tderror = -clamp;
      }
      // 反向传播
      pred.dw[a0] = tderror;
      this.lastG.backward(); // compute gradients on net params

      // update net 更新网络
      R.updateNet(this.net, this.alpha);
      return tderror;
    }
  }

  //////////////////////////////////////////////////////////
  var ActorCriticAgent = function (env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.75);
    this.epsilon = getopt(opt, 'epsilon', 0.1);
    this.c_alpha = getopt(opt, 'c_alpha', 0.01); // Q value learning rate
    this.a_alpha = getopt(opt, 'a_alpha', 0.01); // policy learning rate
    this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0);
    this.num_hidden_units = getopt(opt, 'num_hidden_units', 100);
    this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
    this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
    this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);
    
    this.env = env;
    
    this.exp = []; // experience
    this.expi = 0; // where to insert
    this.t = 0;
    this.state = undefined;
    this.action = undefined;
    this.reward1 = 0;
    // TODO: add your additional variables here
    this.reset();
  }

  ActorCriticAgent.prototype = {
    reset: function () {
      // get state and action feature number
      this.nfs = env.getNumOfStateFeatures();
      this.nfa = env.getMaxNumOfActionFeatures();
      this.nh = this.num_hidden_units;
      this.nha = 20;// 策略网络中间层神经元数
      // you can also get the detail features from env.
      // Critic 网络，输出Q值，它是一层线性网络
      this.aNet = {}; // net只存储了数据，具体数据运算交给Graph, 用来学习参数的。
      this.aNet.W1 = new R.RandMat(this.nha, this.nfs, 0, 0.1);
      this.aNet.b1 = new R.Mat(this.nha, 1, 0, 0.01);
      this.aNet.W2 = new R.RandMat(this.nfa, this.nha, 1, 0.02);
      this.aNet.b2 = new R.Mat(this.nfa, 1, 0, 0.01);

      this.cNet = {}; // fixed Q，固定一段时间参数,Critic
      this.cNet.W1 = new R.RandMat(this.nh, this.nfs, 0, 0.01);
      this.cNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.cNet.W2 = new R.Mat(this.nfa, this.nh, 0, 0.01);
      this.cNet.b2 = new R.Mat(this.nfa, 1, 0, 0.01);
      this.Ps = undefined; // store policy net result matrix
      this.Qs = undefined;

      // 记住最后的网络运算细节
      this.lastCG = undefined; // graph
      this.lastAG = undefined;  // graph

      this.tderror = 0;// TD误差
      this.probs = undefined; // 存储各行为在当前策略下的概率
      this.k = 0;
      this.qsa = 0;
      this.score = 0;
    },

    observe: function () {
      // env should have a function with agent's next state returned
      return this.env.releasedObs();
    },

    policy_based_policy: function(state, needs_backprop=true) {
    // generate an action for agent to execute or observe Q value
      // according to current state.
      // 建立一个列向量，行数为状态数
      var a = undefined;
      var tmat = this.forwardA(state, needs_backprop);
      this.probs = R.softmax(tmat);
      //var s = this.toMatS(state);
      //this.forwardA(this.aNet, s, needs_backprop);
      var r = Math.random();
      var tp = 0.0;
      for (var i = 0; i < this.nfa; i++ ){
        tp += this.probs.w[i];
        if (tp >= r)
          return i;
      }
      return this.probs.w[this.nfa-1];
    },
    random_policy: function(state) {
      return R.randInt(0,this.nfa); 
    },
    // 在CriticActorAgent里，没有调用该函数，而是将其集成值learning里
    performPolicy: function (state, needs_backprop=true) {
      var a = undefined;
      var epsilon = 0.0;
      // epsilon greedy policy 整体上是epsilon贪婪搜索
      if (Math.random() < epsilon) {
        return randInt(0, this.nfa);
      } else {
      return this.policy_based_policy(state, needs_backprop=true);
      }
    },

    performAction: function (action) {
      // execute an action and immediately get a reward from env
      // env should have a method defining its dynamics(model)
      // environment's dynamics should return a reward.
      this.action = action;
      this.reward1 = this.env.dynamics(this.action);
    },

    // 辅助函数，将状态列表转化为forwardQ方法里要求的Mat
    toMatS: function (s) {
      var m_s = new R.Mat(this.nfs, 1);
      m_s.setFrom(s);
      //tmat = this.forward(this.aNet, m_s, false, false)
      return m_s;
    },


    learn: function () {
      // agent lear the model by this.reward1.
      // 学习函数一般都是在已经有了s0
      if (this.c_alpha > 0) {
        var s1 = this.observe(); //得到最新的状态，但不更新到自己的状态中
        this.tderror=  this.learnFromTuple(this.state, this.action, this.reward1, s1);
        //this.tderror = learn_result["td_err"];
        //this.policy_grad = learn_result.policy_grad;
         // decide if we should keep this experience in the replay
        
         if (this.t % this.experience_add_every === 0) {
          this.exp[this.expi] = [this.state, this.action, this.reward1, s1];
          this.expi += 1;
          if (this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
        }
        this.t += 1;

        // sample some additional experience from replay memory and learn from it
        for (var k = 0; k < this.learning_steps_per_iteration; k++) {
          var ri = randInt(0, this.exp.length); // todo: priority sweeps?
          var e = this.exp[ri];
          this.performPolicy(e[0],true);
          //this.action = e[1];
          this.learnFromTuple(e[0], e[1], e[2], e[3])
        }
      }
    },
    forwardC: function(s, needs_backprop) {
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(this.cNet.W1, this.toMatS(s)), this.cNet.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(this.cNet.W2, h1mat), this.cNet.b2);
      if (needs_backprop) {
        this.lastCG = G; // back this up. Kind of hacky isn't it
        this.Qs = a2mat;
      }
      return a2mat;      
    },

    forwardA: function(s, needs_backprop) {
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(this.aNet.W1, this.toMatS(s)), this.aNet.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(this.aNet.W2, h1mat), this.aNet.b2);
      if(needs_backprop) {
        this.lastAG = G; // back this up. Kind of hacky isn't it
        this.Ps = a2mat;
      }
      return a2mat;  
    },
    limitTo: function(value, default_v = 1) {
      var dv = Math.abs(default_v);
      if(value > dv) return dv;
      else if (value < -1 * dv ) return -dv;
      else return value;
    },
    learnFromTuple: function (s0, a0, r1, s1) {

      // 产生a1的策略网络输出不参与反向计算
      var a1 = this.performPolicy(s1, false);
      // 计算s1状态的Q值，该网络输出不参与反向计算
      var q1_mat = this.forwardC(s1, false);
      var q1 = q1_mat.w[a1];// s1,a1对应的Q值
      // 用Q(s1,a1)表示的(s,a)的价值
      var q_prime = r1 + this.gamma * q1; // Q‘（s',a')
      // s0,a0对应的Q值，该网络计算需要反向传播
      var Qs = this.forwardC(s0, true); // Qs === this.Qs;
      //var mean_value = Qs.sum() / Qs.w.length;
      this.qsa = Qs.w[a0];
      var tderror = q_prime - this.qsa;
      this.Qs.dw[a0] = -1 * tderror;
      this.k+=1;
      var mean_Ps_w = 0.0;
      for(var i = 0; i < this.nfa; i++) {
        mean_Ps_w += this.probs.w[i] * this.Ps.w[i];
      }
      this.a_alpha = 0.01;//R.sig(this.k/5e7)-0.45;
      for(var i = a0; i < a0+1; i++) {
        this.score = this.Ps.w[i] - mean_Ps_w;
        this.Ps.dw[i] = -0.01 * this.score * this.qsa/ this.Ps.w[i];
      }

      this.lastCG.backward();
      this.lastAG.backward();

      R.updateNet(this.cNet, this.c_alpha);
      R.updateNet(this.aNet, this.a_alpha);
      // 限制了tderror范围
      /*
      var clamp = this.tderror_clamp;
      if (Math.abs(tderror) > clamp) {  // huber loss to robustify
        if (tderror > clamp) tderror = clamp;
        if (tderror < -clamp) tderror = -clamp;
      }*/
      return tderror;
    }
    // your addtional method here
  }
  //////////////////////////////////////////////////////////

  // 高斯策略的Actor Critic Agent

  //////////////////////////////////////////////////////////
  var GaussianActorCriticAgent = function (env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.75);
    this.epsilon = getopt(opt, 'epsilon', 0.1);
    this.c_alpha = getopt(opt, 'c_alpha', 0.01); // Q value learning rate
    this.a_alpha = getopt(opt, 'a_alpha', 0.01); // policy learning rate
    this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0);
    this.num_hidden_units = getopt(opt, 'num_hidden_units', 100);
    this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
    this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
    this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);

    this.env = env;

    this.exp = []; // experience
    this.expi = 0; // where to insert
    this.t = 0;
    this.state = undefined;
    this.action = undefined;
    this.reward1 = 0;
    // TODO: add your additional variables here
    this.reset();
  }

  GaussianActorCriticAgent.prototype = {
    reset: function () {
      // get state and action feature number
      this.nfs = env.getNumOfStateFeatures();
      this.nfa = env.getMaxNumOfActionFeatures();
      this.nh = this.num_hidden_units;
      // you can also get the detail features from env.
      // Critic 网络，输出Q值，它是一层线性网络
      this.aNet = {}; // net只存储了数据，具体数据运算交给Graph, 用来学习参数的。
      this.aNet.W1 = new R.RandMat(this.nh, this.nfs, 0, 0.01);
      this.aNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.aNet.W2 = new R.OneMat(this.nfa, this.nh, 1, 0.2);
      this.aNet.b2 = new R.Mat(this.nfa, 1, 0, 0.01);

      // critic 网络为输入(s,a) -> 输出 Q型
      this.cNet = {}; // fixed Q，固定一段时间参数,Critic
      this.cNet.W1 = new R.RandMat(this.nh, this.nfs+this.nfa, 0, 0.01);
      this.cNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.cNet.W2 = new R.RandMat(1, this.nh, 0, 0.01);
      this.cNet.b2 = new R.Mat(1, 1, 0, 0.01);
      this.Ps = undefined; // store policy net result matrix
      this.Qs = undefined;

      this.cNetHiddenOut = undefined;
      this.aNetHiddenOut = undefined;

      // 记住最后的网络运算细节
      this.lastCG = undefined; // graph
      this.lastAG = undefined;  // graph
      this.std = 0.05; // 高斯采样的标准差
      this.tderror = 0;// 可视化TD误差
      this.qsa = 0; // 可视化
    },

    observe: function () {
      // env should have a function with agent's next state returned
      return this.env.releasedObs();
    },
    random_policy: function(state, needs_backprop = true) {
      var a = new Array(2);
      a[0] = R.random(-1,+1);
      a[1] = R.random(-1,1);
      return a;
    },
    policy_based_policy: function (state, needs_backprop = true) {
      // generate an action for agent to execute or observe Q value
      // according to current state.
      var a = new Array(2);
      var tmat = this.forwardA(state, needs_backprop);
      a = [0, 0];
      var std = 1;
      if (this.t > 1e6) {
        std = 1 / ((1.00 * this.t) / 1e5 + 1);
      }
      a[0] = R.randNorm(tmat.w[0], std);
      a[1] = R.randNorm(tmat.w[1], std);
      if (Math.abs(a[0]) > 1000 || Math.abs(a[1]) > 1000) {
        console.log("Caution");
      }
      // 获得网络输出向量最大值的索引进而货的行为
      return a;
    },
    // 在CriticActorAgent里，没有调用该函数，而是将其集成值learning里
    performPolicy: function (state, needs_backprop = true) {
        return this.policy_based_policy(state, needs_backprop);
    },

    performAction: function (action) {
      // execute an action and immediately get a reward from env
      // env should have a method defining its dynamics(model)
      // environment's dynamics should return a reward.
      this.action = action;
      this.reward1 = this.env.dynamics(this.action);
    },

    // 辅助函数，将状态列表转化为forwardQ方法里要求的Mat
    toMatS: function (s) {
      var m_s = new R.Mat(this.nfs, 1);
      m_s.setFrom(s);
      //tmat = this.forward(this.aNet, m_s, false, false)
      return m_s;
    },


    learn: function () {
      // agent lear the model by this.reward1.
      // 学习函数一般都是在已经有了s0
      if (this.c_alpha > 0) {
        var s1 = this.observe(); //得到最新的状态，但不更新到自己的状态中
        this.tderror = this.learnFromTuple(this.state, this.action, this.reward1, s1);
        // decide if we should keep this experience in the replay
        if (this.t % this.experience_add_every === 0) {
          this.exp[this.expi] = [this.state, this.action, this.reward1, s1];
          this.expi += 1;
          if (this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
        }
        this.t += 1;

        if (this.t > 1e6){
          // sample some additional experience from replay memory and learn from it
          for (var k = 0; k < this.learning_steps_per_iteration; k++) {
            var ri = randInt(0, this.exp.length); // todo: priority sweeps?
            var e = this.exp[ri];
            agent.performPolicy(e[0]);
            this.learnFromTuple(e[0], e[1], e[2], e[3])
          }
        }
      }
    },
    // 要重写，输入是s+a长度
    forwardC: function (s, a, needs_backprop) {
      var G = new R.Graph(needs_backprop);
      var input = new R.Mat(this.nfs+this.nfa, 1);
      input.setFrom(s);
      for (var i = 0; i< this.nfa; i++) {
        input.w[this.nfs+i] = a[i];
      }
      //W1: 100 * 10, s+a: 10 * 1
      var a1mat = G.add(G.mul(this.cNet.W1, input), this.cNet.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(this.cNet.W2, h1mat), this.cNet.b2);
      if (needs_backprop) {
        this.lastCG = G; // back this up. Kind of hacky isn't it
        this.cNetHiddenOut = h1mat;
        this.Qs = a2mat;
      }
      return a2mat;
    },

    forwardA: function (s, needs_backprop) {
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(this.aNet.W1, this.toMatS(s)), this.aNet.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(this.aNet.W2, h1mat), this.aNet.b2);
      if (needs_backprop) {
        this.lastAG = G; // back this up. Kind of hacky isn't it
        this.aNetHiddenOut = h1mat;
        this.Ps = a2mat;
      }
      return a2mat;
    },
    limitTo: function (value, default_v = 1) {
      var dv = Math.abs(default_v);
      if (value > dv) return dv;
      else if (value < -1 * dv) return -dv;
      else return value;
    },
    learnFromTuple: function (s0, a0, r1, s1) {

      // 产生a1的策略网络输出不参与反向计算
      var a1 = this.performPolicy(s1, false);
      // 计算s1状态的Q值，该网络输出不参与反向计算
      var q1_mat = this.forwardC(s1, a1, false); // Q(s1,a1)
      var q1 = q1_mat.w[0];// 应该只有一个元素
      // 用Q(s1,a1)表示的(s,a)的价值
      var q_prime = r1 + this.gamma * q1; // Q‘（s',a')
      // s0,a0对应的Q值，该网络计算需要反向传播
      var Qs = this.forwardC(s0, a0, true); // Qs === this.Qs;
      var qsa = Qs.w[0];
      this.qsa = qsa;
      var tderror = q_prime - qsa;
      this.Qs.dw[0] = -1 * tderror;

      for (var i = a0; i < a0+1; i++) {
        var grad = -1 * this.qsa * (a0[i] - this.Ps.w[i])//;(this.std*this.std);
        //grad = this.limitTo(grad, 1);
        this.Ps.dw[i] = grad;
      }
      this.lastCG.backward();
      this.lastAG.backward();

      R.updateNet(this.cNet, this.c_alpha);
      R.updateNet(this.aNet, this.a_alpha);
      // 限制了tderror范围
      /*
      var clamp = this.tderror_clamp;
      if (Math.abs(tderror) > clamp) {  // huber loss to robustify
        if (tderror > clamp) tderror = clamp;
        if (tderror < -clamp) tderror = -clamp;
      }*/
      return tderror;
    }
    // your addtional method here
  }
  //////////////////////////////////////////////////////////





  // buggy implementation, doesnt work...
  var SimpleReinforceAgent = function (env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.75); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  SimpleReinforceAgent.prototype = {
    reset: function () {
      this.nfs = this.env.getNumStates();
      this.nfa = this.env.getMaxNumActions();
      this.nh = 100; // number of hidden units
      this.nhb = 100; // and also in the baseline lstm

      this.actorNet = {};
      this.actorNet.W1 = new R.RandMat(this.nh, this.nfs, 0, 0.01);
      this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.actorNet.W2 = new R.RandMat(this.nfa, this.nh, 0, 0.1);
      this.actorNet.b2 = new R.Mat(this.nfa, 1, 0, 0.01);
      this.actorOutputs = [];
      this.actorGraphs = [];
      this.actorActions = []; // sampled ones

      this.rewardHistory = [];

      this.baselineNet = {};
      this.baselineNet.W1 = new R.RandMat(this.nhb, this.nfs, 0, 0.01);
      this.baselineNet.b1 = new R.Mat(this.nhb, 1, 0, 0.01);
      this.baselineNet.W2 = new R.RandMat(this.nfa, this.nhb, 0, 0.01);
      this.baselineNet.b2 = new R.Mat(this.nfa, 1, 0, 0.01);
      this.baselineOutputs = [];
      this.baselineGraphs = [];

      this.t = 0;
    },
    forwardActor: function (s, needs_backprop) {
      var net = this.actorNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return { 'a': a2mat, 'G': G }
    },
    forwardValue: function (s, needs_backprop) {
      var net = this.baselineNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return { 'a': a2mat, 'G': G }
    },
    act: function (slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.nfs, 1);
      s.setFrom(slist);

      // forward the actor to get action output
      var ans = this.forwardActor(s, true);
      var amat = ans.a;
      var ag = ans.G;
      this.actorOutputs.push(amat);
      this.actorGraphs.push(ag);

      // forward the baseline estimator
      var ans = this.forwardValue(s, true);
      var vmat = ans.a;
      var vg = ans.G;
      this.baselineOutputs.push(vmat);
      this.baselineGraphs.push(vg);

      // sample action from the stochastic gaussian policy
      var a = R.copyMat(amat);
      var gaussVar = 0.02;
      a.w[0] = R.randNorm(0, gaussVar);
      a.w[1] = R.randNorm(0, gaussVar);

      this.actorActions.push(a);

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    learn: function (r1) {
      // perform an update on Q function
      this.rewardHistory.push(r1);
      var n = this.rewardHistory.length;
      var baselineMSE = 0.0;
      var nup = 100; // what chunk of experience to take
      var nuse = 80; // what chunk to update from
      if (n >= nup) {
        // lets learn and flush
        // first: compute the sample values at all points
        var vs = [];
        for (var t = 0; t < nuse; t++) {
          var mul = 1;
          // compute the actual discounted reward for this time step
          var V = 0;
          for (var t2 = t; t2 < n; t2++) {
            V += mul * this.rewardHistory[t2];
            mul *= this.gamma;
            if (mul < 1e-5) { break; } // efficiency savings
          }
          // get the predicted baseline at this time step
          var b = this.baselineOutputs[t].w[0];
          for (var i = 0; i < this.nfa; i++) {
            // [the action delta] * [the desirebility]
            var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
            if (update > 0.1) { update = 0.1; }
            if (update < -0.1) { update = -0.1; }
            this.actorOutputs[t].dw[i] += update;
          }
          var update = - (V - b);
          if (update > 0.1) { update = 0.1; }
          if (update < 0.1) { update = -0.1; }
          this.baselineOutputs[t].dw[0] += update;
          baselineMSE += (V - b) * (V - b);
          vs.push(V);
        }
        baselineMSE /= nuse;
        // backprop all the things
        for (var t = 0; t < nuse; t++) {
          this.actorGraphs[t].backward();
          this.baselineGraphs[t].backward();
        }
        R.updateNet(this.actorNet, this.alpha); // update actor network
        R.updateNet(this.baselineNet, this.beta); // update baseline network

        // flush
        this.actorOutputs = [];
        this.rewardHistory = [];
        this.actorActions = [];
        this.baselineOutputs = [];
        this.actorGraphs = [];
        this.baselineGraphs = [];

        this.tderror = baselineMSE;
      }
      this.t += 1;
      this.r0 = r1; // store for next update
    },
  }

  // buggy implementation as well, doesn't work
  var RecurrentReinforceAgent = function (env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  RecurrentReinforceAgent.prototype = {
    reset: function () {
      this.nfs = this.env.getNumStates();
      this.nfa = this.env.getMaxNumActions();
      this.nh = 40; // number of hidden units
      this.nhb = 40; // and also in the baseline lstm

      this.actorLSTM = R.initLSTM(this.nfs, [this.nh], this.nfa);
      this.actorG = new R.Graph();
      this.actorPrev = null;
      this.actorOutputs = [];
      this.rewardHistory = [];
      this.actorActions = [];

      this.baselineLSTM = R.initLSTM(this.nfs, [this.nhb], 1);
      this.baselineG = new R.Graph();
      this.baselinePrev = null;
      this.baselineOutputs = [];

      this.t = 0;

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
    },
    act: function (slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.nfs, 1);
      s.setFrom(slist);

      // forward the LSTM to get action distribution
      var actorNext = R.forwardLSTM(this.actorG, this.actorLSTM, [this.nh], s, this.actorPrev);
      this.actorPrev = actorNext;
      var amat = actorNext.o;
      this.actorOutputs.push(amat);

      // forward the baseline LSTM
      var baselineNext = R.forwardLSTM(this.baselineG, this.baselineLSTM, [this.nhb], s, this.baselinePrev);
      this.baselinePrev = baselineNext;
      this.baselineOutputs.push(baselineNext.o);

      // sample action from actor policy
      var gaussVar = 0.05;
      var a = R.copyMat(amat);
      for (var i = 0, n = a.w.length; i < n; i++) {
        a.w[0] += R.randNorm(0, gaussVar);
        a.w[1] += R.randNorm(0, gaussVar);
      }
      this.actorActions.push(a);

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;
      return a;
    },
    learn: function (r1) {
      // perform an update on Q function
      this.rewardHistory.push(r1);
      var n = this.rewardHistory.length;
      var baselineMSE = 0.0;
      var nup = 100; // what chunk of experience to take
      var nuse = 80; // what chunk to also update
      if (n >= nup) {
        // lets learn and flush
        // first: compute the sample values at all points
        var vs = [];
        for (var t = 0; t < nuse; t++) {
          var mul = 1;
          var V = 0;
          for (var t2 = t; t2 < n; t2++) {
            V += mul * this.rewardHistory[t2];
            mul *= this.gamma;
            if (mul < 1e-5) { break; } // efficiency savings
          }
          var b = this.baselineOutputs[t].w[0];
          // todo: take out the constants etc.
          for (var i = 0; i < this.nfa; i++) {
            // [the action delta] * [the desirebility]
            var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
            if (update > 0.1) { update = 0.1; }
            if (update < -0.1) { update = -0.1; }
            this.actorOutputs[t].dw[i] += update;
          }
          var update = - (V - b);
          if (update > 0.1) { update = 0.1; }
          if (update < 0.1) { update = -0.1; }
          this.baselineOutputs[t].dw[0] += update;
          baselineMSE += (V - b) * (V - b);
          vs.push(V);
        }
        baselineMSE /= nuse;
        this.actorG.backward(); // update params! woohoo!
        this.baselineG.backward();
        R.updateNet(this.actorLSTM, this.alpha); // update actor network
        R.updateNet(this.baselineLSTM, this.beta); // update baseline network

        // flush
        this.actorG = new R.Graph();
        this.actorPrev = null;
        this.actorOutputs = [];
        this.rewardHistory = [];
        this.actorActions = [];

        this.baselineG = new R.Graph();
        this.baselinePrev = null;
        this.baselineOutputs = [];

        this.tderror = baselineMSE;
      }
      this.t += 1;
      this.r0 = r1; // store for next update
    },
  }

  // Currently buggy implementation, doesnt work
  var DeterministPG = function (env, opt) {
    this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
    this.epsilon = getopt(opt, 'epsilon', 0.5); // for epsilon-greedy policy
    this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
    this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
    this.env = env;
    this.reset();
  }
  DeterministPG.prototype = {
    reset: function () {
      this.nfs = this.env.getNumStates();
      this.nfa = this.env.getMaxNumActions();
      this.nh = 100; // number of hidden units

      // actor
      this.actorNet = {};
      this.actorNet.W1 = new R.RandMat(this.nh, this.nfs, 0, 0.01);
      this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
      this.actorNet.W2 = new R.RandMat(this.nfa, this.nfs, 0, 0.1);
      this.actorNet.b2 = new R.Mat(this.nfa, 1, 0, 0.01);
      this.ntheta = this.nfa * this.nfs + this.nfa; // number of params in actor

      // critic
      this.criticw = new R.RandMat(1, this.ntheta, 0, 0.01); // row vector

      this.r0 = null;
      this.s0 = null;
      this.s1 = null;
      this.a0 = null;
      this.a1 = null;
      this.t = 0;
    },
    forwardActor: function (s, needs_backprop) {
      var net = this.actorNet;
      var G = new R.Graph(needs_backprop);
      var a1mat = G.add(G.mul(net.W1, s), net.b1);
      var h1mat = G.tanh(a1mat);
      var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
      return { 'a': a2mat, 'G': G }
    },
    act: function (slist) {
      // convert to a Mat column vector
      var s = new R.Mat(this.nfs, 1);
      s.setFrom(slist);

      // forward the actor to get action output
      var ans = this.forwardActor(s, false);
      var amat = ans.a;
      var ag = ans.G;

      // sample action from the stochastic gaussian policy
      var a = R.copyMat(amat);
      if (Math.random() < this.epsilon) {
        var gaussVar = 0.02;
        a.w[0] = R.randNorm(0, gaussVar);
        a.w[1] = R.randNorm(0, gaussVar);
      }
      var clamp = 0.25;
      if (a.w[0] > clamp) a.w[0] = clamp;
      if (a.w[0] < -clamp) a.w[0] = -clamp;
      if (a.w[1] > clamp) a.w[1] = clamp;
      if (a.w[1] < -clamp) a.w[1] = -clamp;

      // shift state memory
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s;
      this.a1 = a;

      return a;
    },
    utilJacobianAt: function (s) {
      var ujacobian = new R.Mat(this.ntheta, this.nfa);
      for (var a = 0; a < this.nfa; a++) {
        R.netZeroGrads(this.actorNet);
        var ag = this.forwardActor(this.s0, true);
        ag.a.dw[a] = 1.0;
        ag.G.backward();
        var gflat = R.netFlattenGrads(this.actorNet);
        ujacobian.setColumn(gflat, a);
      }
      return ujacobian;
    },
    learn: function (r1) {
      // perform an update on Q function
      //this.rewardHistory.push(r1);
      if (!(this.r0 == null)) {
        var Gtmp = new R.Graph(false);
        // dpg update:
        // first compute the features psi:
        // the jacobian matrix of the actor for s
        var ujacobian0 = this.utilJacobianAt(this.s0);
        // now form the features \psi(s,a)
        var psi_sa0 = Gtmp.mul(ujacobian0, this.a0); // should be [this.ntheta x 1] "feature" vector
        var qw0 = Gtmp.mul(this.criticw, psi_sa0); // 1x1
        // now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
        var ujacobian1 = this.utilJacobianAt(this.s1);
        var ag = this.forwardActor(this.s1, false);
        var psi_sa1 = Gtmp.mul(ujacobian1, ag.a);
        var qw1 = Gtmp.mul(this.criticw, psi_sa1); // 1x1
        // get the td error finally
        var tderror = this.r0 + this.gamma * qw1.w[0] - qw0.w[0]; // lol
        if (tderror > 0.5) tderror = 0.5; // clamp
        if (tderror < -0.5) tderror = -0.5;
        this.tderror = tderror;

        // update actor policy with natural gradient
        var net = this.actorNet;
        var ix = 0;
        for (var p in net) {
          var mat = net[p];
          if (net.hasOwnProperty(p)) {
            for (var i = 0, n = mat.w.length; i < n; i++) {
              mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
              ix += 1;
            }
          }
        }
        // update the critic parameters too
        for (var i = 0; i < this.ntheta; i++) {
          var update = this.beta * tderror * psi_sa0.w[i];
          this.criticw.w[i] += update;
        }
      }
      this.r0 = r1; // store for next update
    },
  }

  // exports
  global.DPAgent = DPAgent;
  global.TDAgent = TDAgent;
  global.DQNAgent = DQNAgent;
  global.ActorCriticAgent = ActorCriticAgent;
  global.GaussianActorCriticAgent = GaussianActorCriticAgent;
  //global.SimpleReinforceAgent = SimpleReinforceAgent;
  //global.RecurrentReinforceAgent = RecurrentReinforceAgent;
  //global.DeterministPG = DeterministPG;
})(RL);



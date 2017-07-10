var R = {}; // a tool library

(function(global){
  "use strict";

  //Utility fun
  function assert(condition, message) {
    if(!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message;
    }
  }

  var isVReturned = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(isVReturned) {
      isVReturned = false;
      return v_val;
    }
    var u = 2 * Math.random() - 1;
    var v = 2 * Math.random() - 1;
    var r = u*u + v*v;
    if( r==0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2 * Math.log(r)/r);
    v_val = v*c;
    isVReturned = true;
    return u*c;
  }

  // 几种产生随机数的方式
  // a,b 之间的浮点数
  var random = function(a, b) { return Math.random()*(b-a)+a; }
  // a,b 之间的整数
  var randInt = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
  // 从以mu为均值，std为标准差的正态分布中随机采样
  var randNorm = function(mu, std){ return mu+gaussRandom()*std; }

  // helper function returns array of zeros of length n
  // and uses typed arrays if available
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i] = 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
  }

  // Mat holds a matrix
  var Mat = function(n,d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  }
  Mat.prototype = {
    get: function(row, col) { 
      // slow but careful accessor function
      // we want row-major order
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      return this.w[ix];
    },
    set: function(row, col, v) {
      // slow but careful accessor function
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      this.w[ix] = v; 
    },
    setFrom: function(arr) {
      for(var i=0,n=arr.length;i<n;i++) {
        this.w[i] = arr[i]; 
      }
    },
    setColumn: function(m, i) {
      for(var q=0,n=m.w.length;q<n;q++) {
        this.w[(this.d * q) + i] = m.w[q];
      }
    },
    toJSON: function() {
      var json = {};
      json['n'] = this.n;
      json['d'] = this.d;
      json['w'] = this.w;
      return json;
    },
    fromJSON: function(json) {
      this.n = json.n;
      this.d = json.d;
      this.w = zeros(this.n * this.d);
      this.dw = zeros(this.n * this.d);
      for(var i=0,n=this.n * this.d;i<n;i++) {
        this.w[i] = json.w[i]; // copy over weights
      }
    }
  }

  var copyMat = function(b) {
    var a = new Mat(b.n, b.d);
    a.setFrom(b.w);
    return a;
  }

  var copyNet = function(net) {
    // nets are (k,v) pairs with k = string key, v = Mat()
    var new_net = {};
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        new_net[p] = copyMat(net[p]);
      }
    }
    return new_net;
  }

  var updateMat = function(m, alpha) {
    // updates in place
    for(var i=0,n=m.n*m.d;i<n;i++) {
      if(m.dw[i] !== 0) {
        m.w[i] += - alpha * m.dw[i];
        m.dw[i] = 0;
      }
    }
  }

  var updateNet = function(net, alpha) {
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        updateMat(net[p], alpha);
      }
    }
  }

  var netToJSON = function(net) {
    var j = {};
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        j[p] = net[p].toJSON();
      }
    }
    return j;
  }
  var netFromJSON = function(j) {
    var net = {};
    for(var p in j) {
      if(j.hasOwnProperty(p)){
        net[p] = new Mat(1,1); // not proud of this
        net[p].fromJSON(j[p]);
      }
    }
    return net;
  }
  var netZeroGrads = function(net) {
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        var mat = net[p];
        gradFillConst(mat, 0);
      }
    }
  }
  var netFlattenGrads = function(net) {
    var n = 0;
    for(var p in net) { if(net.hasOwnProperty(p)){ var mat = net[p]; n += mat.dw.length; } }
    var g = new Mat(n, 1);
    var ix = 0;
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        var mat = net[p];
        for(var i=0,m=mat.dw.length;i<m;i++) {
          g.w[ix] = mat.dw[i];
          ix++;
        }
      }
    }
    return g;
  }

  // return Mat but filled with random numbers from gaussian
  var RandMat = function(n,d,mu,std) {
    var m = new Mat(n, d);
    fillRandn(m,mu,std);
    //fillRand(m,-std,std); // kind of :P
    return m;
  }

  // Mat utils
  // fill matrix with random gaussian numbers
  var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randNorm(mu, std); } }
  var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = random(lo, hi); } }
  var gradFillConst = function(m, c) { for(var i=0,n=m.dw.length;i<n;i++) { m.dw[i] = c } }

// Transformer definitions
  var Graph = function(needs_backprop) {
    if(typeof needs_backprop === 'undefined') { needs_backprop = true; }
    this.needs_backprop = needs_backprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    this.backprop = [];
  }
  Graph.prototype = {
    backward: function() {
      for(var i=this.backprop.length-1;i>=0;i--) {
        this.backprop[i](); // tick!
      }
    },
    rowPluck: function(m, ix) {
      // pluck a row of m with index ix and return it as col vector
      assert(ix >= 0 && ix < m.n);
      var d = m.d;
      var out = new Mat(d, 1);
      for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=d;i<n;i++){ m.dw[d * ix + i] += out.dw[i]; }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    tanh: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = Math.tanh(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    sigmoid: function(m) {
      // sigmoid nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = sig(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    relu: function(m) {
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = Math.max(0, m.w[i]); // relu
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    mul: function(m1, m2) {
      // multiply matrices m1 * m2
      assert(m1.d === m2.n, 'matmul dimensions misaligned');

      var n = m1.n;
      var d = m2.d;
      var out = new Mat(n,d);
      for(var i=0;i<m1.n;i++) { // loop over rows of m1
        for(var j=0;j<m2.d;j++) { // loop over cols of m2
          var dot = 0.0;
          for(var k=0;k<m1.d;k++) { // dot product loop
            dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
          }
          out.w[d*i+j] = dot;
        }
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<m1.n;i++) { // loop over rows of m1
            for(var j=0;j<m2.d;j++) { // loop over cols of m2
              for(var k=0;k<m1.d;k++) { // dot product loop
                var b = out.dw[d*i+j];
                m1.dw[m1.d*i+k] += m2.w[m2.d*k+j] * b;
                m2.dw[m2.d*k+j] += m1.w[m1.d*i+k] * b;
              }
            }
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    add: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] + m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += out.dw[i];
            m2.dw[i] += out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    dot: function(m1, m2) {
      // m1 m2 are both column vectors
      assert(m1.w.length === m2.w.length);
      var out = new Mat(1,1);
      var dot = 0.0;
      for(var i=0,n=m1.w.length;i<n;i++) {
        dot += m1.w[i] * m2.w[i];
      }
      out.w[0] = dot;
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += m2.w[i] * out.dw[0];
            m2.dw[i] += m1.w[i] * out.dw[0];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    eltmul: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] * m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += m2.w[i] * out.dw[i];
            m2.dw[i] += m1.w[i] * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
  }

  var softmax = function(m) {
      var out = new Mat(m.n, m.d); // probability volume
      var maxval = -999999;
      for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

      var s = 0.0;
      for(var i=0,n=m.w.length;i<n;i++) { 
        out.w[i] = Math.exp(m.w[i] - maxval);
        s += out.w[i];
      }
      for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

      // no backward pass here needed
      // since we will use the computed probabilities outside
      // to set gradients directly on m
      return out;
    }

  var Solver = function() {
    this.decay_rate = 0.999;
    this.smooth_eps = 1e-8;
    this.step_cache = {};
  }
  Solver.prototype = {
    step: function(model, step_size, regc, clipval) {
      // perform parameter update
      var solver_stats = {};
      var num_clipped = 0;
      var num_tot = 0;
      for(var k in model) {
        if(model.hasOwnProperty(k)) {
          var m = model[k]; // mat ref
          if(!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
          var s = this.step_cache[k];
          for(var i=0,n=m.w.length;i<n;i++) {

            // rmsprop adaptive learning rate
            var mdwi = m.dw[i];
            s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

            // gradient clip
            if(mdwi > clipval) {
              mdwi = clipval;
              num_clipped++;
            }
            if(mdwi < -clipval) {
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
      solver_stats['ratio_clipped'] = num_clipped*1.0/num_tot;
      return solver_stats;
    }
  }

  var initLSTM = function(input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
      var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      var hidden_size = hidden_sizes[d];

      // gates parameters
      model['Wix'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Wih'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bi'+d] = new Mat(hidden_size, 1);
      model['Wfx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Wfh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bf'+d] = new Mat(hidden_size, 1);
      model['Wox'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Woh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bo'+d] = new Mat(hidden_size, 1);
      // cell write params
      model['Wcx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Wch'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bc'+d] = new Mat(hidden_size, 1);
    }
    // decoder params
    model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
    model['bd'] = new Mat(output_size, 1);
    return model;
  }

  var forwardLSTM = function(G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    if(prev == null || typeof prev.h === 'undefined') {
      var hidden_prevs = [];
      var cell_prevs = [];
      for(var d=0;d<hidden_sizes.length;d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d],1)); 
        cell_prevs.push(new R.Mat(hidden_sizes[d],1)); 
      }
    } else {
      var hidden_prevs = prev.h;
      var cell_prevs = prev.c;
    }

    var hidden = [];
    var cell = [];
    for(var d=0;d<hidden_sizes.length;d++) {

      var input_vector = d === 0 ? x : hidden[d-1];
      var hidden_prev = hidden_prevs[d];
      var cell_prev = cell_prevs[d];

      // input gate
      var h0 = G.mul(model['Wix'+d], input_vector);
      var h1 = G.mul(model['Wih'+d], hidden_prev);
      var input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

      // forget gate
      var h2 = G.mul(model['Wfx'+d], input_vector);
      var h3 = G.mul(model['Wfh'+d], hidden_prev);
      var forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

      // output gate
      var h4 = G.mul(model['Wox'+d], input_vector);
      var h5 = G.mul(model['Woh'+d], hidden_prev);
      var output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

      // write operation on cells
      var h6 = G.mul(model['Wcx'+d], input_vector);
      var h7 = G.mul(model['Wch'+d], hidden_prev);
      var cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

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
    var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

    // return cell memory, hidden representation and output
    return {'h':hidden, 'c':cell, 'o' : output};
  }

  var sig = function(x) {
    // helper function for computing sigmoid
    return 1.0/(1+Math.exp(-x));
  }

  var maxi = function(w) {
    // argmax of array w
    var maxv = w[0];
    var maxix = 0;
    for(var i=1,n=w.length;i<n;i++) {
      var v = w[i];
      if(v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  }

  var samplei = function(w) {
    // sample argmax from w, assuming w are 
    // probabilities that sum to one
    var r = random(0,1);
    var x = 0.0;
    var i = 0;
    while(true) {
      x += w[i];
      if(x > r) { return i; }
      i++;
    }
    // return w.length - 1; // pretty sure we should never get here?
  }

  // various utils
  global.assert = assert;
  global.zeros = zeros;
  global.maxi = maxi;
  global.samplei = samplei;
  global.randInt = randInt;
  global.randNorm = randNorm;
  global.softmax = softmax;
  // classes
  global.Mat = Mat;
  global.RandMat = RandMat;
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


var RL = {};

(function(global){
  "use strict";
  // 从R对象来的功能
  var assert = R.assert;

  // 自己写的一些功能函数
  var isArrayEqual = function(arr1, arr2){
    if (arr1 === undefined && arr2 === undefined)
      return false;
    if (arr1 === undefined || arr2 === undefined)
      return false;
    if (arr1.length !== arr2.length){
      return false;
    }
    for(var i = 0; i < arr1.length; i++){
      if (arr1[i] !== arr2[i]) // correct for number and string.
        return false;
    }
    return true;
  }

  // 向量，由特征组成
  var Vector = function(values=undefined) {
    this.values = values;
  }

  Vector.prototype = {
    // 判断两个向量是否完全相同
    isEqualTo: function(vector) {
      var t1 = typeof(this.values);
      var t2 = typeof(vector.values);
      if (t1 !== t2)
        return false;
      else if(t1 === "number"){
        return this.values === vector.values;
      } else {
        return isArrayEqual(this.values, vector.values);
      }
    }
  }

  var State = function(values) {
    this.values = values;
  }
  State.prototype = new Vector(this.values)

  var Action = function(values) {
    //this.name = name;
    this.values = values;
  }
  Action.prototype = new Vector(this.values)

  var Agent = function(env){
    //this.Qsa = {};
    //this.Qs = {};
    //this.Es = {};
    //this.Esa = {};
    this.curPolicy = undefined;
    this.stateFeatures = undefined; //个体的状态空间特征
    this.actionFeatures = undefined;//个体的行为空间特征

    this.model = undefined;
    this.env = env;
  }

  Agent.prototype = {
    loadActions: function(){
      env.loadActions();      
    },
    observe: function(){
      return env.releaseInfo()
    },
    act: function(a){
      env.model(a)
    },
    performPolicy: function(policy, state) {
      var action = new Action("",this.action_features,undefined)

      return action;
    },
    learning: function(learn_fun, param) {
      learn_fun(param); //用param提示的参数进行learn_fun学习
      // param可以使json结构，learn_fun是具体的学习算法，它从param
      // 里提取需要的参数。如果参数没有提取到，则取该算法的默认值
    }
  }


  var Env = function() {
    this.stateFeatures = undefined; //环境的状态空间特征
    this.actionFeatures = undefined;//环境提供的行为空间特征
    this.startState = undefined; // Episode开始状态只有一个
    this.endStates = undefined;  // Episode终止状态可以有多个
    this.agentState = undefined // Agent 当前状态，初始设置为开始状态
    this.init();
  }
  Env.prototype = {
    init: function() {
      // 初始化环境
      this.endStates = [];
      console.log("init environment");
    },
    reset: function() {
      // 重置环境
      console.log("reset environment");
      this.init();
    },
    model: function(s_t, a) {
      var s_t1 = undefined;
      var reward_t1 = undefined;
      // compute s_t1 and reward_t1
      return {"s_t1":s_t1,
              "reward_t1":reward_t1
            };
    },
    releaseInfo: function() {
      // 整理过滤释放给Agent的信息
    },
    isEndState: function(s) {
      for (var i = 0; i < this.endStates.length; i++){
        if(s.isEqualTo(this.endStates[i]))
          return true;
      }
      return false;
    },
    isAgentInEndState: function() {
      // 判断是否进入Episode终止状态
      return this.isEndState(this.agentState);
    }
  }



  var GridWorld = function() {
    this.stateFeatures=["index"];
    this.actionFeatures=["left","right","up","down"];
    this.rwdArr = undefined;
    this.T = undefined;
    this.reset();
  }
  GridWorld.prototype = new Env();
  GridWorld.prototype.init = function () {
    this.startState = new State(0);
    this.endStates = new Array();
    this.endStates.push(new State([55]));
  
    this.h = 10;
    this.w = 10
    this.area = this.h * this.w;

    var rwd_arr = R.zeros(this.area);
    var T = R.zeros(this.area)
    rwd_arr[55] = 1;

    rwd_arr[54] = -1;
    //rwd_arr[63] = -1;
    rwd_arr[64] = -1;
    rwd_arr[65] = -1;
    rwd_arr[85] = -1;
    rwd_arr[86] = -1;

    rwd_arr[37] = -1;
    rwd_arr[33] = -1;
    //rwd_arr[77] = -1;
    rwd_arr[67] = -1;
    rwd_arr[57] = -1;

    // make some cliffs
    for (var q = 0; q < 8; q++) { var off = (q + 1) * this.h + 2; T[off] = 1; rwd_arr[off] = 0; }
    for (var q = 0; q < 6; q++) { var off = 4 * this.h + q + 2; T[off] = 1; rwd_arr[off] = 0; }
    T[5 * this.h + 2] = 0; 
    rwd_arr[5 * this.h + 2] = 0; // make a hole
    this.rwdArr = rwd_arr;
    this.T = T;
  };
  GridWorld.prototype.reset = GridWorld.prototype.init;

  GridWorld.prototype.model = function(s_t,a) {
    var reward_t1 = this.rwdArr[s_t];
    var s_t1 = GridWorld.prototype.nextStateDistribution(s_t,a);
    return {"s_t1":s_t1,
            "reward_t1":reward_t1
          }
  }
  GridWorld.prototype.nextStateDistribution = function(s_t, a) {
    var s_t1 = undefined;
    if(this.T === 1){
      //几乎不可能，初始化时小心即可
    } else if (this.isEndState(s_t)) {
      s_t1 = s_t;
    }
  },
  GridWorld.prototype.isAgentInEndState = function() {

  } 
  // as a class of RL
  global.Agent = Agent;
  global.Env = Env;
  global.Vector = Vector;
  global.State = State;
  global.Action = Action;
  global.GridWorld = GridWorld;
})(RL);

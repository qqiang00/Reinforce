var RL = {};

(function(global){
  "use strict";

  // 随机数函数
  var random = function (a, b) { return Math.random() * (b - a) + a; }
  var randInt = function (a, b) { return Math.floor(Math.random() * (b - a) + a); }
  var randNorm = function (mu, std) { return mu + gaussRandom() * std; }

  // 生成长度为n元素值均为0的数组
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

  // 从R对象来的功能
  var assert = function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  // 判断两个数组的内容是否完全相同
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

  //状态是一个向量
  var State = function(values) {
    this.values = values;
  }
  State.prototype = new Vector(this.values)

  //行为也是一个向量
  var Action = function(values) {
    //this.name = name;
    this.values = values;
  }
  Action.prototype = new Vector(this.values)

  // =====================================
  // 个体（智能体）
  // =====================================

  var Agent = function(env){
    //this.Qsa = {};
    //this.Qs = {};
    //this.Es = {};
    //this.Esa = {};
    this.curPolicy = undefined; // 正在使用的策略，策略是一个函数
    this.bacPolicy = undefined; // 备用策略，策略是一个函数
    this.stateFeatures = undefined; //个体的状态空间特征
    this.actionFeatures = undefined;//个体的行为空间特征

    this.model = undefined;   // 模型
    this.env = env;   // 环境，个体必须接受env参数才能生成
  }

  Agent.prototype = {
    loadActions: function(){
      env.loadActions();      
    },
    // 观测环境
    observe: function(){
      return env.releaseInfo()
    },
    // 执行一个行为
    act: function(a){
      env.model(a)
    },
    // 执行策略
    performPolicy: function(policy, state) {
      var action = new Action("",this.action_features,undefined)

      return action;
    },

    // 策略评估
    policyEvaluate: function () {

    },

    updatePolicy: function() {

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

    model: function(s, a) {
      var s1 = undefined;
      var reward1 = undefined;
      // compute s_t1 and reward_t1
      return {"s_t1":s1,
              "reward_t1":reward1
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
    this.stateFeatures=["x","y"]; // 次序即是Vector的次序
    this.actionFeatures=["left","right","up","down"]; // 次序即是Vector的次序
    this.gridFeatures = ["normal","wall"];  // 格子类型：正常或是不可通过的障碍
    this.R = undefined; // 离开某状态的即时奖励
    this.T = undefined;

    this.h = 10;
    this.w = 10;
    this.area = this.h * this.w;
    this.reset();

    this.startState = undefined;
    this.endStates = undefined;
    this.agentState = undefined;
  }
  GridWorld.prototype = new Env();
  GridWorld.prototype.init = function () {

    this.h = 10;
    this.w = 10
    this.area = this.h * this.w;

    var rwd_arr = zeros(this.area);
    var T = zeros(this.area)

    this.RS = rwd_arr;
    this.T = T;
  };

  GridWorld.prototype.reset = GridWorld.prototype.init;

  GridWorld.prototype.nextStateDistribution = function(s_t, a) {
    var s_t1 = undefined;
    if(this.T === 1){
      //几乎不可能，初始化时小心即可
    } else if (this.isEndState(s_t)) {
      s_t1 = s_t;
    }
  }

  GridWorld.prototype.getNextState = function(s, a) {

  }
  GridWorld.prototype.rewardOfState = function(s) {

  }

  GridWorld.prototype.model = function(s,a) {
    var reward1 = GridWorld.prototype.rewardOfState(s);
    var s1 = GridWorld.prototype.getNextState(s,a);
    return {"s_t1":s1,
            "reward_t1":reward1
          }
  }
  // 将位置pos的格子设置为feature描述的格子。
  GridWorld.prototype.setGridTo = function (pos, feature) {
    this.T
  }
  // as a class of RL
  global.Agent = Agent;
  global.Env = Env;
  global.Vector = Vector;
  global.State = State;
  global.Action = Action;
  global.GridWorld = GridWorld;
})(RL);

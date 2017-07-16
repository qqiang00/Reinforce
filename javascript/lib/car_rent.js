// ==================== 帮助函数 =================
// 计算阶乘
var factorial = function (n) {
    if (n < 0) {
        return -1;
    } else if (n === 0 || n === 1) {
        return 1;
    } else {
        return (n * factorial(n - 1))
    }
}

// 计算恰好n时的泊松分布概率
var poisson = function (n, lambda) {
    return Math.exp(-1 * lambda) * Math.pow(lambda, n) / factorial(n)
}

// 计算所有小于n的泊松分布概率之和（不包括n）
var poissonLessThan = function (n, lambda) {
    var p = 0.0;
    for (var i = 0; i < n; i++) {
        p += poisson(i, lambda);
    }
    return p;
}

// 计算所有不小于n的泊松概率之和
var poissonNoLessThan = function (n, lambda) {
    return 1.00 - poissonLessThan(n, lambda);
}

// 验证泊松分布概率函数
var _verify_poisson = function (lambda) {
    var n = lambda * 4;
    var total = 0;
    for (var i = 0; i < n; i++) {
        total += poisson(i, lambda)
    }
    return total;// should be very close to 1.00
}

// 获取opt对象里的参数，以field_name为属性，没有时用缺省值
var getopt = function (opt, field_name, default_value) {
    if (typeof opt === 'undefined') { return default_value; }
    return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
}

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

//======================= 环境 =======================

var CarRent = function () {
    this.Rarr = null;
    this.l1_out = 3;    //地点1的借出泊松分布参数,y坐标
    this.l1_in = 3;     //地点1的归还
    this.l2_out = 4;    //地点2的借出
    this.l2_in = 2;     //地点2的归还
    this.rwd_per_car = 10.0;//每借出一辆车的奖励
    this.max_move = 5.0;    //最多移动5辆车
    
    
    this.reset();
}

CarRent.prototype = {
    //stox: function (s) { return Math.floor(s / this.gh); },
    //stoy: function (s) { return s % this.gh; },
    //xytos: function (x, y) { return x * this.gh + y; },

    stox: function (s) { return s % this.gw; },
    stoy: function (s) { return this.gh - 1 - Math.floor(s / this.gw); },
    xytos: function (x, y) { return x + this.gw * (this.gh - 1 - y); },

    reset: function () {
        this.gh = 21;   // 地点1的最大车辆数-1
        this.gw = 21;   // 地点2的最大车辆数-1
        this.gs = this.gh * this.gw;
        //初始化状态奖励
        var Rarr = new Array(this.gs);
        for (var x = 0; x < this.gw; x++) {// loc2相当于x
            for (var y = 0; y < this.gh; y++) {//loc1相当于y
                var rwd = this.reward_per_loc(y, this.l1_out, this.l1_in);
                rwd += this.reward_per_loc(x, this.l2_out, this.l2_in);
                Rarr[this.xytos(x, y)] = rwd;
            }
        }
        this.Rarr = Rarr;
    },
    // 某一个租赁点，有ct辆车，同时一定的接触和归还分布是的奖励期望
    reward_per_loc: function (ct, l_out, l_in, per_car = 10.0, max_cars = 20) {
        var reward = 0;
        for (var i = 0; i <= max_cars; i++) {
            //可以100%概率借出,此时奖励取决于实际借车数
            if (i <= ct)
                reward += poisson(i, l_out) * i;
            else {
                //当借出车辆多余现有车辆时，概率需要结合
                reward += poisson(i, l_out) * poissonNoLessThan((i - ct), l_in) * i;
            }
        }
        return reward * per_car;
    },
    // 计算状态即时奖励
    reward: function (s, a, ns) {
        return this.Rarr[s];
    },
    // 状态转移，a可以取[-5，+5]，+5表示从yloc1)移动车辆到x(loc2)
    nextStateDistribution: function (s, a_) {
        var a = a_ - 5; // a是数组索引，-5变成实际移动车辆数
        var x = this.stox(s);
        var y = this.stoy(s);
        var newy = 0;
        var newx = 0;
        //先计算最少、最多能从y移动多少辆车
        var min_move = Math.max(-5, y - (this.gw - 1), -1 * x);
        var max_move = Math.min(y, this.gh - 1 - x, 5);
        var act_move = a;//实际移动车辆数
        if (act_move < min_move)
            act_move = min_move;
        if (act_move > max_move)
            act_move = max_move;
        newx = x + act_move;
        newy = y - act_move;
        return this.xytos(newx, newy);
    },
    // 个体允许的行为
    allowedActions: function (s) {
        var actions = []
        var x = this.stox(s);
        var y = this.stoy(s);
        var min_move = Math.max(-5, y - (this.gw - 1), -1 * x);
        var max_move = Math.min(y, this.gh - 1 - x, 5);
        //for (var i = -1 * this.max_move; i <= this.max_move; i++){
        for (var i = min_move; i <= max_move; i++) {
            actions.push(i + 5);
        }
        return actions;
    },
    // 辅助函数，与环境相关
    randomState: function () { return Math.floor(Math.random() * (this.gs)); },
    startState: function () { return this.randomState(); },
    getNumStates: function () { return this.gs; },
    getMaxNumActions: function () { return 11; },
}


//======================= 个体代理人 ==================

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
        this.totalStateNum = this.env.getNumStates();
        this.totalActionNum = this.env.getMaxNumActions();
        this.V = zeros(this.totalStateNum);
        this.P = zeros(this.totalStateNum * this.totalActionNum);
        // initialize uniform random policy
        for (var s = 0; s < this.totalStateNum; s++) {
            // 在环境模型里过滤了不能操作的行为
            var poss = this.env.allowedActions(s);
            //var poss = this.env.allActions(s);
            for (var i = 0, n = poss.length; i < n; i++) {
                this.P[poss[i] * this.totalStateNum + s] = 1.0 / poss.length;
                //this.P[poss[5]*this.totalStateNum+s]=1.0;
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
            var prob = this.P[a * this.totalStateNum + s];
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
        var Vnew = zeros(this.totalStateNum);
        for (var s = 0; s < this.totalStateNum; s++) {
            // integrate over actions in a stochastic policy
            // note that we assume that policy probability mass over allowed actions sums to one
            var v = 0.0;
            var poss = this.env.allowedActions(s);
            //var poss = this.env.allActions(s);
            for (var i = 0, n = poss.length; i < n; i++) {
                var a = poss[i];
                var prob = this.P[a * this.totalStateNum + s]; // probability of taking action under policy
                if (prob === 0) { continue; } // no contribution, skip for speed
                var ns = this.env.nextStateDistribution(s, a);
                var rs = this.env.reward(s, a, ns); // reward for s->a->ns transition
                v += prob * (rs + this.gamma * this.V[ns]);
            }
            Vnew[s] = Math.floor(v);

        }
        this.V = Vnew; // swap
    },
    updatePolicy: function () {
        // update policy to be greedy w.r.t. learned Value function
        for (var s = 0; s < this.totalStateNum; s++) {
            //var poss = this.env.allActions(s);
            var poss = this.env.allowedActions(s);
            // compute value of taking each allowed action
            var vmax, nmax;
            var vs = [];
            for (var i = 0, n = poss.length; i < n; i++) {
                var a = poss[i];
                var ns = this.env.nextStateDistribution(s, a);
                var rs = this.env.reward(s, a, ns);
                var v = Math.floor(rs + this.gamma * this.V[ns]);
                vs.push(v);
                if (i === 0 || v > vmax) { vmax = v; nmax = 1; }
                else if (v === vmax) { nmax += 1; }
            }
            // update policy smoothly across all argmaxy actions
            for (var i = 0, n = poss.length; i < n; i++) {
                var a = poss[i];
                this.P[a * this.totalStateNum + s] = (vs[i] === vmax) ? 1.0 / nmax : 0.0;
            }
        }
    },
}




//======================= UI绘制 ====================



var rs = {};
var trs = {};
var tvs = {};
//var pas = {};
var cs = 36;  // cell size
var initGrid = function () {
    var d3elt = d3.select('#draw');
    d3elt.html('');
    rs = {};
    trs = {};
    tvs = {};
    //pas = {};

    var gh = env.gh; // height in cells
    var gw = env.gw; // width in cells
    var gs = env.gs; // total number of cells

    var w = 800;
    var h = 800;
    svg = d3elt.append('svg').attr('width', w).attr('height', h)
        .append('g').attr('transform', 'scale(1)');

    // define a marker for drawing arrowheads
    svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("refX", 3)
        .attr("refY", 2)
        .attr("markerWidth", 3)
        .attr("markerHeight", 4)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M 0,0 V 4 L3,2 Z");

    for (var y = 0; y < gh; y++) {
        for (var x = 0; x < gw; x++) {
            var xcoord = 25 + x * cs;
            var ycoord = h - 50 - y * cs;
            var s = env.xytos(x, y);

            var g = svg.append('g');
            // click callbackfor group
            g.on('click', function (ss) {
                return function () { cellClicked(ss); } // close over s
            }(s));

            // set up cell rectangles
            var r = g.append('rect')
                .attr('x', xcoord)
                .attr('y', ycoord)
                .attr('height', cs)
                .attr('width', cs)
                .attr('fill', '#FFF')
                .attr('stroke', 'black')
                .attr('stroke-width', 2);
            rs[s] = r;

            // reward text
            var tr = g.append('text')
                .attr('x', xcoord + 2)
                .attr('y', ycoord + 30)
                .attr('font-size', 10)
                .text('');
            trs[s] = tr;

            // skip rest for cliffs
            // if (env.T[s] === 1) { continue; }

            // value text
            var tv = g.append('text')
                .attr('x', xcoord + 2)
                .attr('y', ycoord + 15)
                .attr('font-size', 10)
                .text('');
            tvs[s] = tv;

            // policy arrows
            /*pas[s] = []
            for (var a = 0; a < 4; a++) {
                var pa = g.append('line')
                    .attr('x1', xcoord)
                    .attr('y1', ycoord)
                    .attr('x2', xcoord)
                    .attr('y2', ycoord)
                    .attr('stroke', 'black')
                    .attr('stroke-width', '2')
                    .attr("marker-end", "url(#arrowhead)");
                pas[s].push(pa);
            }*/
        }
    }

}
// 在存储0-1的数组中找到最大值的索引
var maxIndex = function (arr) {
    var max = -1;
    var index = -1;
    for (var i = 0; i < arr.length; i++) {
        if (arr[i] > max) {
            max = arr[i];
            index = i;
        }
    }
    return index;
}

var drawGrid = function () {
    var gh = env.gh; // height in cells
    var gw = env.gw; // width in cells
    var gs = env.gs; // total number of cells

    // updates the grid with current state of world/agent
    for (var y = 0; y < gh; y++) {
        for (var x = 0; x < gw; x++) {
            //var xcoord = x * cs;
            //var ycoord = y * cs;
            var r = 255, g = 255, b = 255;
            var s = env.xytos(x, y);

            var vv = agent.V[s];
            var maxa = -1;
            var maxp = 0;
            for (var i = 0; i < env.getMaxNumActions() / 2; i++) {
                var a = 5 + i;//5为不移动
                var prob = agent.P[a * gs + s];
                if (prob > maxp + 0.001) {
                    maxp = prob;
                    maxa = a;
                }
                a = 5 - i;
                var prob = agent.P[a * gs + s];
                if (prob > maxp + 0.001) {
                    maxp = prob;
                    maxa = a;
                }
            }
            //found maxa
            var ms = 50.0;

            if (maxa >= 5) { g = 255; r = 255 - (maxa - 5) * ms; b = 255 - (maxa - 5) * ms; }
            if (maxa < 5) { g = 255 - (5 - maxa) * ms; r = 255; b = 255 - (5 - maxa) * ms; }
            var vcol = 'rgb(' + Math.floor(r) + ',' + Math.floor(g) + ',' + Math.floor(b) + ')';

            // update colors of rectangles based on value
            var r = rs[s];
            if (s === selected) {
                // highlight selected cell
                r.attr('fill', '#FF0');
            } else {
                r.attr('fill', vcol);
            }

            // write reward texts
            var rv = env.Rarr[s];
            var tr = trs[s];
            if (rv !== 0) {
                tr.text(maxa - 5 + " " + env.Rarr[s].toFixed(1));
            }

            // skip rest for cliff
            //if (env.T[s] === 1) continue;

            // write value
            var tv = tvs[s];
            //tv.text(maxa-5);
            tv.text(agent.V[s].toFixed(2));

        }
    }
}

//======================= 其它页面响应 ===============
var selected = -1;
var cellClicked = function (s) {
    if (s === selected) {
        selected = -1; // toggle off
        $("#creward").html('(选择一个格子)');
    } else {
        selected = s;
        $("#creward").html(env.Rarr[s].toFixed(2));
        $("#rewardslider").slider('value', env.Rarr[s]);
    }
    drawGrid(); // redraw
}

var updatePolicy = function () {
    agent.updatePolicy();
    drawGrid();
}

var evaluatePolicy = function () {
    agent.evaluatePolicy()
    drawGrid();
}

var pid = -1;
var evaluatePolicy2 = function () {
    if (pid === -1) {
        pid = setInterval(function () {
            agent.evaluatePolicy();
            //agent.updatePolicy();
            drawGrid();
        }, 100);
    } else {
        clearInterval(pid);
        pid = -1;
    }
}

var sid = -1;
var runValueIteration = function () {
    if (sid === -1) {
        sid = setInterval(function () {
            agent.evaluatePolicy();
            agent.updatePolicy();
            drawGrid();
        }, 100);
    } else {
        clearInterval(sid);
        sid = -1;
    }
}

function resetAll() {
    env.reset();
    agent.reset();
    drawGrid();
}

// ==================== 主入口 body onload调用===============
var agent, env;
function start() {
    env = new CarRent(); // create environment
    agent = new DPAgent(env, { 'gamma': 0.9 }); // create an agent, yay!
    initGrid();
    drawGrid();

    $("#rewardslider").slider({
        min: -5,
        max: 5.1,
        value: 0,
        step: 0.1,
        slide: function (event, ui) {
            if (selected >= 0) {
                env.Rarr[selected] = ui.value;
                $("#creward").html(ui.value.toFixed(2));
                drawGrid();
            } else {
                $("#creward").html('(选择一个格子)');
            }
        }
    });
    // suntax highlighting
    //marked.setOptions({highlight:function(code){ return hljs.highlightAuto(code).value; }});
    $(".md").each(function () {
        $(this).html(marked($(this).html()));
    });
    renderJax();
}

var jaxrendered = false;
function renderJax() {
    if (jaxrendered) { return; }
    (function () {
        var script = document.createElement("script");
        script.type = "text/javascript";
        script.src = "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";
        document.getElementsByTagName("head")[0].appendChild(script);
        jaxrendered = true;
    })();
};

//=================== 完 ==================
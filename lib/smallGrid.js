// 格子世界，属于一种环境
var Gridworld = function () {
    this.Rarr = null; // 奖励数组一维
    this.T = null; // 格子类型数组(可以表示为空格，不能进入等等)
    this.reset()
}

Gridworld.prototype = {
    reset: function () {

        //格子长宽
        this.gh = 4;
        this.gw = 4;
        // 格子总数，也是总的状态数
        this.gs = this.gh * this.gw;

        // specify some rewards
        var Rarr = new Array(this.gs);
        var T = new Array(this.gs);
        // 左上角和右下角奖励为0，设为终止位置
        for (var i = 0; i < this.gs; i++) {
            Rarr[i] = -1; //暂时设定所有格子奖励为-1
            T[i] = 0; //所有格子都是可以进入的。
        }
        // 左上角右下角格子奖励为0
        Rarr[0] = 0;
        Rarr[this.gs - 1] = 0;
        this.Rarr = Rarr;
        this.T = T;
    },

    // 奖励函数
    reward: function (s, a, ns) {
        // reward of being in s, taking action a, and ending up in ns
        return this.Rarr[s];
    },

    // 根据当前状态和行为决定下一个状态
    nextStateDistribution: function (s, a) {
        var ns = s;
        // given (s,a) return distribution over s' (in sparse form)
        if (this.T[s] === 1) {
            // cliff! oh no!
            // var ns = 0; // reset to state zero (start)
        } else if (s === 0 || s === 15) {
            // 出于终止状态的格子停留在原地。
            ns = s;
        } else {
            // 先把格子状态表示为2维形式，根据行为计算后再返回1维
            var nx, ny;
            var x = this.stox(s);
            var y = this.stoy(s);
            if (a === 0) { nx = x - 1; ny = y; }
            if (a === 1) { nx = x; ny = y - 1; }
            if (a === 2) { nx = x; ny = y + 1; }
            if (a === 3) { nx = x + 1; ny = y; }
            // 格子不能越界
            if (nx < 0) nx = 0;
            else if (nx >= this.gw) nx = this.gw - 1;
            if (ny < 0) ny = 0;
            else if (ny >= this.gh) ny = this.gh - 1;
            ns = nx * this.gh + ny;
            // 如果格子进了不能进的地方，则退回原地
            if (this.T[ns] === 1) {
                ns = s;
            }
        }
        // 返回一个确定性的状态
        return ns;
    },

    sampleNextState: function (s, a) {
        var ns = this.nextStateDistribution(s, a);
        var r = this.Rarr[s];
        r -= 1;
        var out = { 'ns': ns, 'r': r };
        if (s === 15 && ns === 0) {
            // episode is over
            out.reset_episode = true;
        }
        return out;
    },

    allActions: function (s) {
        return [0, 1, 2, 3];
    },

    allowedActions: function (s) {
        // 允许所有行为，与格子所处位置无关。
        return [0, 1, 2, 3];
    },
    // 随机状态不包括0和15
    randomState: function () { return Math.floor(1 + Math.random() * (this.gs - 1)); },
    startState: function () { return this.randomState(); },
    getNumStates: function () { return this.gs; },
    getMaxNumActions: function () { return 4; },

    // private functions
    stox: function (s) { return Math.floor(s / this.gh); },
    stoy: function (s) { return s % this.gh; },
    xytos: function (x, y) { return x * this.gh + y; },
}

// ------
// 绘制接口
// ------
var rs = {};
var trs = {};
var tvs = {};
var pas = {};
var cs = 60;  // cell size
var initGrid = function () {
    var d3elt = d3.select('#draw');
    d3elt.html('');
    rs = {};
    trs = {};
    tvs = {};
    pas = {};

    var gh = env.gh; // height in cells
    var gw = env.gw; // width in cells
    var gs = env.gs; // total number of cells

    var w = 300;
    var h = 300;
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
            var xcoord = x * cs;
            var ycoord = y * cs;
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
                .attr('x', xcoord + 5)
                .attr('y', ycoord + 55)
                .attr('font-size', 10)
                .text('');
            trs[s] = tr;

            // skip rest for cliffs
            if (env.T[s] === 1) { continue; }

            // value text
            var tv = g.append('text')
                .attr('x', xcoord + 5)
                .attr('y', ycoord + 20)
                .text('');
            tvs[s] = tv;

            // policy arrows
            pas[s] = []
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
            }
        }
    }

}

var drawGrid = function () {
    var gh = env.gh; // height in cells
    var gw = env.gw; // width in cells
    var gs = env.gs; // total number of cells

    // updates the grid with current state of world/agent
    for (var y = 0; y < gh; y++) {
        for (var x = 0; x < gw; x++) {
            var xcoord = x * cs;
            var ycoord = y * cs;
            var r = 255, g = 255, b = 255;
            var s = env.xytos(x, y);

            var vv = agent.V[s];
            var ms = 100;
            if (vv > 0) { g = 255; r = 255 - vv * ms; b = 255 - vv * ms; }
            if (vv < 0) { g = 255 + vv * ms; r = 255; b = 255 + vv * ms; }
            var vcol = 'rgb(' + Math.floor(r) + ',' + Math.floor(g) + ',' + Math.floor(b) + ')';
            if (env.T[s] === 1) { vcol = "#AAA"; rcol = "#AAA"; }

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
                tr.text('R ' + rv.toFixed(1))
            }

            // skip rest for cliff
            if (env.T[s] === 1) continue;

            // write value
            var tv = tvs[s];
            tv.text(agent.V[s].toFixed(2));

            // update policy arrows
            var paa = pas[s];
            for (var a = 0; a < 4; a++) {
                var pa = paa[a];
                var prob = agent.P[a * gs + s];
                if (prob === 0) { pa.attr('visibility', 'hidden'); }
                else { pa.attr('visibility', 'visible'); }
                var ss = cs / 2 * prob * 0.9;
                if (a === 0) { nx = -ss; ny = 0; }
                if (a === 1) { nx = 0; ny = -ss; }
                if (a === 2) { nx = 0; ny = ss; }
                if (a === 3) { nx = ss; ny = 0; }
                pa.attr('x1', xcoord + cs / 2)
                    .attr('y1', ycoord + cs / 2)
                    .attr('x2', xcoord + cs / 2 + nx)
                    .attr('y2', ycoord + cs / 2 + ny);
            }
        }
    }
}

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
    agent.V[0] = 0;
    agent.V[15]= 0;
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

var agent, env;
function start() {
    env = new Gridworld(); // create environment
    agent = new RL.DPAgent(env, { 'gamma': 1.00 }); // create an agent, yay!
    initGrid();
    drawGrid();

    $("#rewardslider").slider({
        min: -5,
        max: 5.1,
        value: 0,
        step: 0.1,
        slide: function(event, ui) {
          if(selected >= 0) {
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
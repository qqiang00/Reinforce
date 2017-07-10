
// 位置对象，记录水平、垂直位置索引，整数。
var Pos = function(x,y){
    this.x = x;
    this.y = y;
}

Pos.prototype = {
    add: function(p) { return new Pos(this.x + p.x, this.y + p.y); },
    sub: function(p) { return new Pos(this.x - p.x, this.y - p.y); }
}



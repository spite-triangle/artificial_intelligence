!function () {
    "use strict";
    var generateSpan = function (className, content) {
        return `<span class="${className}">${content}</span>`;
    };
    
    // 复制按钮 
    var btn = ' <i class="fa fa-clipboard term_btn" data-command="text"></i> ';

    // 按钮提示
    window.$docsify.plugins = [].concat(window.$docsify.plugins, function (hook, vm) {
        // 处理路径
        var dealPath = function (content) {
            // 划分用户名与路径
            var parts = content.split(':');
            return generateSpan('term_user', parts[0]) + ':' + generateSpan('term_path', parts[1]);
        };

        // 处理指令
        var dealCommand = function (content) {
            // 划分用户名与路径
            var parts = content.split('//');
            if (parts.length > 1) {
                return generateSpan('term_command', parts[0].trim()) + btn.replace('text', parts[0].trim()) + generateSpan('term_annotation', '//' + parts[1]);
            } else {
                return generateSpan('term_command', parts[0].trim()) + btn.replace('text', parts[0].trim());
            }
        };
        var getElementByAttr = function (tag, dataAttr, reg) {
            var aElements = document.getElementsByTagName(tag);
            var aEle = [];
            for (var i = 0; i < aElements.length; i++) {
                var ele = aElements[i].getAttribute(dataAttr);
                if (reg.test(ele)) {
                    aEle.push(aElements[i]);
                }
            }
            return aEle;
        };

        hook.doneEach(function () {
            // 查找所有的term
            var pres = getElementByAttr('pre', 'data-lang', /term/);

            // 遍历所有的term
            pres.forEach(function (pre, index) {

                // 修改 pre 标签
                pre.setAttribute('class', 'window_term');

                // 修改 code 标签
                // 拆分行
                var lines = pre.childNodes[0].innerHTML.split('\n');
                var convertedText = "";
                var i = 0;
                for (i; i < lines.length; i++) {
                    var parts = lines[i].split('$ ');
                    // 结果还是指令
                    if (parts.length > 1) {
                        // 指令
                        convertedText = convertedText + dealPath(parts[0].trim()) + "$ " + dealCommand(parts[1].trim()) + '</br>';
                    } else {
                        // 结果
                        convertedText = convertedText + lines[i].trim() + '</br>';
                    }
                }

                pre.innerHTML = '<div class="cycles"> <div class="cycle term_cycle1"> </div> <div class="cycle term_cycle2"> </div> <div class="cycle term_cycle3"> </div>  </div> <code class="lang_term">' + convertedText + '</code>';
            });
        });

    });

    // 按钮提示
    window.$docsify.plugins = [].concat(window.$docsify.plugins, function (hook, vm) {
        hook.doneEach(function () {
            // 获取所有的指令按钮
            var term_btns = document.querySelectorAll('.term_btn');
            // 遍历所有按钮
            term_btns.forEach(function (btn,index){
                btn.addEventListener('click', function(e){

                    // 防止多次点击
                    if(e.target.className == 'fa fa-thumbs-up term_btn') return;

                    // 将指令复制到剪贴板
                    var aux = document.createElement("input"); 
                    aux.setAttribute("value", e.target.getAttribute("data-command")); 
                    document.body.appendChild(aux); 
                    aux.select();
                    document.execCommand("copy"); 
                    document.body.removeChild(aux);

                    // 修改图标
                    e.target.setAttribute('class','fa fa-thumbs-up term_btn');

                    // 定时器改回来
                    setTimeout(()=> {
                        e.target.setAttribute('class','fa fa-clipboard term_btn');
                    },500);

                });
            });
        });

    });
}();
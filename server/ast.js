const { parse, find } = require('abstract-syntax-tree');
const sourcemap = require('abstract-syntax-tree/src/sourcemap');
var fs = require("fs");

fs.readFile("source.txt", (err, data) => {
    if (err) throw err;
  
    const tree = parse(data.toString());
    console.log(find(tree, [name="48150"]))
  });

// fs.readFile("C:/Users/Hadiy/OneDrive/Desktop/webpage-crawler-extension/server/source.txt", function(source){
//     console.log(source.toString());
//     // const tree = parse(source);
//     // // console.log(tree);
// });


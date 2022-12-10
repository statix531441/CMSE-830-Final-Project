---
type: Mini-Project
number: 2
due: 
---
- [ ] #task #Submit CMSE 830 Mini-Project 02 📅 null

course:: [[CMSE 830]]
notes:: [[CMSE 830 Notes#2022-10-22]]

---

```dataviewjs

const { promisify } = require('util');
const exec = promisify(require('child_process').exec);
const fs = require("fs");
const path = require("path");
const bigroot = "D:/Sids Vault/"

const folder = bigroot + dv.current().file.folder.toString();
const files = await fs.promises.readdir(folder);

let already_exists = false

for (let file of files){
	if (!file.contains(".ipynb_") && file.contains(".ipynb")){
		already_exists = file
		break
	}
}
if (already_exists){
	already_exists = folder.concat("/",already_exists).toString()
	already_exists = already_exists.split("/").join("\\")
}

let notebook = folder +
				"/" +
				dv.current().file.name +
				" - Siddharth Ashok Unnithan.ipynb"




const buttonMaker = (notebook, already_exists) => {

	let cmd1 = `jupyter-lab "${notebook}"`;
	let cmd2 = `copy "D:\\Sids Vault\\templates\\jupyter_notebook_template.json" "${notebook}"`
	let cmd3 = `copy "${already_exists}" "${notebook}"`
	
    const btn = this.container.createEl('button', {"text": "Open Jupyter Notebook"});
    btn.addEventListener('click', async (evt) => {
        evt.preventDefault();
        
		if(fs.existsSync(notebook)){
			await exec(cmd1);
		}
		else{
			if(already_exists){
		        await exec(cmd3);
		        await exec(cmd1);
	        }
	        else{
				await exec(cmd2);
				await exec(cmd1);
			}
		}
	
    });
    return btn;
}

let btn = buttonMaker(notebook, already_exists);

```

```dataviewjs

const { promisify } = require('util');
const exec = promisify(require('child_process').exec);
const fs = require("fs");
const path = require("path");
const bigroot = "D:/Sids Vault/"

const folder = bigroot + dv.current().file.folder.toString();
const files = await fs.promises.readdir(folder);

let already_exists = false

for (let file of files){
	if (!file.contains(".ipynb_") && file.contains(".xopp")){
		already_exists = file
		break
	}
}
if (already_exists){
	already_exists = folder.concat("/",already_exists).toString()
	already_exists = already_exists.split("/").join("\\")
}

let notebook = folder +
				"/" +
				dv.current().file.name +
				" - Siddharth Ashok Unnithan.xopp"


notebook = notebook.split("/").join("\\")

const buttonMaker = (notebook, already_exists, template) => {

	let cmd1 = `"${notebook}"`;
	let cmd2 = `copy "${template}" "${notebook}"`
	let cmd3 = `copy "${already_exists}" "${notebook}"`
	
    const btn = this.container.createEl('button', {"text": "Open Xournal Notebook"});
    btn.addEventListener('click', async (evt) => {
        evt.preventDefault();
        
		if(fs.existsSync(notebook)){
			await exec(cmd1);
		}
		else{
			if(already_exists){
		        await exec(cmd3);
		        await exec(cmd1);
	        }
	        else{
				await exec(cmd2);
				await exec(cmd1);
			}
		}
	
    });
    return btn;
}
let template = "D:\\Sids Vault\\templates\\xournal_notebook_template.xopp"
let btn = buttonMaker(notebook, already_exists, template);

dv.span(btn)

```

- [ ] From the correlation matrix we can drop atemp and count ig.

---
### Goal

To develop a business model for bike rental services.

#### Ideas

- [ ] Make new col called weekends ?
- [ ] Fluctuating prices could be a potential marketing strategy to convert casual riders to registered ones.
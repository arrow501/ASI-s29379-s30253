// Rotate an orthonormal viewing basis in an ambient principal-component plane.
// Projection is Z @ B. This is a rigid Givens rotation, never point interpolation.
export function rotateBasis(basis, i, j, radians) {
  if (i === j) throw new Error("Choose two distinct principal components.");
  const next = basis.map(row => [...row]);
  const c = Math.cos(radians), s = Math.sin(radians);
  for (let axis = 0; axis < 2; axis++) {
    next[i][axis] = c * basis[i][axis] - s * basis[j][axis];
    next[j][axis] = s * basis[i][axis] + c * basis[j][axis];
  }
  return next;
}

export function project(scores, basis) {
  return scores.map(row => [0, 1].map(axis =>
    row.reduce((sum, score, pc) => sum + score * basis[pc][axis], 0)));
}

function render({model, el}) {
  el.classList.add("pca-tour");
  el.innerHTML = `
    <div class="tour-heading"><div><span class="tour-kicker">ORTHOGONAL TOUR</span>
      <h3>One cloud. Moving axes.</h3></div><span class="tour-badge" data-role="variance"></span></div>
    <div class="tour-controls">
      <label>Rotation plane <select aria-label="First rotation component" data-role="first"></select></label>
      <span>↔</span><select aria-label="Second rotation component" data-role="second"></select>
      <button data-role="play">▶ Play</button><button data-role="reset">Reset PC1 × PC2</button>
      <button data-role="select" aria-pressed="false">Box select</button>
      <button data-role="clear">Clear selection</button>
      <button data-role="fit">Fit all points</button>
      <label>Colour <select data-role="colour" aria-label="Colour points by">
        <option value="outcome">Outcome</option><option value="chronology">Row order</option>
        <option value="job">Job</option><option value="contact">Contact</option>
        <option value="poutcome">Previous outcome</option></select></label>
    </div>
    <div class="tour-angle"><label>Angle in selected plane <input data-role="angle" type="range"
      min="-180" max="180" step="0.1" value="0" aria-label="Rotation angle"></label>
      <output data-role="degrees">0°</output><label>Speed <input data-role="speed" type="range"
      min="5" max="60" value="20" aria-label="Animation degrees per second"></label></div>
    <div class="tour-stage"><canvas tabindex="0" aria-label="PCA projection: drag or use arrow keys to rotate; box select to inspect a region"></canvas>
      <div class="tour-tooltip" hidden></div></div>
    <div class="tour-legend"></div><div class="tour-formula"></div>
    <p class="tour-status" aria-live="polite"></p>
    <p class="tour-hint">Drag to rotate • scroll to zoom • hover to inspect • box-select a region.
      Arrow keys rotate; Space plays/pauses. Camera scale stays fixed while rotating.</p>`;
  const get = role => el.querySelector(`[data-role="${role}"]`);
  const canvas = el.querySelector("canvas"), ctx = canvas.getContext("2d");
  const tip = el.querySelector(".tour-tooltip");
  const scores = model.get("scores"), records = model.get("records"), variance = model.get("variance");
  const d = scores[0]?.length ?? 0;
  if (d < 2) {el.textContent = "At least two PCA dimensions are needed."; return;}
  const identity = () => Array.from({length:d}, (_, i) => [Number(i===0), Number(i===1)]);
  let basis = identity(), angle = 0, playing = false, frame = null, lastTime = null;
  let drag = null, box = null, selectionMode = false, selected = new Set(model.get("selected_ids"));
  let pixels = [], width = 900, height = 510;
  const norms = scores.map(row => Math.hypot(...row)).sort((a,b) => a-b);
  const radius = Math.max(norms[Math.floor(norms.length * .98)], .01);
  let zoom = 1;
  const palette = ["#2a9d8f", "#e9c46a", "#8d86c9", "#e76f51", "#67b7dc", "#b8c480", "#d69ac4", "#ad9b83", "#788b99", "#a1c9b8", "#f7b27a", "#b2b2c9"];
  const categories = {};
  for (const field of ["job", "contact", "poutcome"]) categories[field] = [...new Set(records.map(r => String(r[field])))].sort();
  for (const select of [get("first"), get("second")]) {
    for (let pc=0; pc<d; pc++) {const opt=document.createElement("option");opt.value=pc;opt.textContent=`PC${pc+1}`;select.append(opt);}
  }
  get("first").value = "0"; get("second").value = String(d > 2 ? 2 : 1);
  const syncSelection = () => {model.set("selected_ids", [...selected].sort((a,b)=>a-b));model.save_changes();};
  function rotate(delta) {
    const i=Number(get("first").value), j=Number(get("second").value);
    if (i===j) {get("second").value=String((i+1)%d);}
    basis=rotateBasis(basis, Number(get("first").value), Number(get("second").value), delta*Math.PI/180);
    angle=((angle+delta+540)%360)-180;
    get("angle").value=angle;get("degrees").textContent=`${angle.toFixed(1)}°`;
    draw();
  }
  function colour(record) {
    const field=get("colour").value;
    if (field==="outcome") return record.outcome==="yes" ? "#f4a340" : "#7e95ac";
    if (field==="chronology") return `hsl(${255-record.chronology*210},65%,55%)`;
    return palette[categories[field].indexOf(String(record[field]))%palette.length];
  }
  function draw() {
    const ratio=window.devicePixelRatio || 1;
    canvas.width=width*ratio;canvas.height=height*ratio;
    ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
    const scale=Math.min(width,height)*.43/radius*zoom;
    const coordinates=project(scores,basis);
    pixels=coordinates.map(([x,y])=>[width/2+x*scale,height/2-y*scale]);
    ctx.strokeStyle="rgba(128,149,170,.25)";ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(0,height/2);ctx.lineTo(width,height/2);ctx.moveTo(width/2,0);ctx.lineTo(width/2,height);ctx.stroke();
    // Draw positives last for visibility; coordinates and selection IDs never reorder.
    const order=records.map((_,i)=>i).sort((a,b)=>Number(records[a].outcome==="yes")-Number(records[b].outcome==="yes"));
    for (const index of order) {
      const [x,y]=pixels[index];
      ctx.globalAlpha=selected.size && !selected.has(index) ? .12 : .73;
      ctx.fillStyle=colour(records[index]);ctx.beginPath();ctx.arc(x,y,selected.has(index)?4.5:2.8,0,Math.PI*2);ctx.fill();
      if (selected.has(index)) {ctx.strokeStyle="#fff";ctx.stroke();}
    }
    ctx.globalAlpha=1;
    if (box) {ctx.fillStyle="rgba(42,157,143,.12)";ctx.strokeStyle="#2a9d8f";ctx.fillRect(box.x,box.y,box.w,box.h);ctx.strokeRect(box.x,box.y,box.w,box.h);}
    const captured=basis.reduce((sum,row,i)=>sum+variance[i]*(row[0]**2+row[1]**2),0);
    get("variance").textContent=`${(100*captured).toFixed(1)}% total variance in view`;
    const expression = axis => basis.map((row,i)=>({weight:row[axis],pc:i+1})).filter(p=>Math.abs(p.weight)>.025)
      .map(p=>`${p.weight>=0?"+":"−"}${Math.abs(p.weight).toFixed(2)} PC${p.pc}`).join(" ").replace(/^\+/,"");
    el.querySelector(".tour-formula").textContent=`Horizontal: ${expression(0)}   |   Vertical: ${expression(1)}`;
    const offscreen=pixels.filter(([x,y])=>x<0||x>width||y<0||y>height).length;
    el.querySelector(".tour-status").textContent=`${scores.length.toLocaleString()} sampled rows · ${selected.size} selected · ${offscreen} off-screen · ${d} fixed PCA dimensions · BᵀB = I`;
    const field=get("colour").value;
    const legend=el.querySelector(".tour-legend");legend.replaceChildren();
    if(field==="outcome"||field==="chronology") {
      legend.textContent=field==="outcome" ? "● blue-grey: no   ● amber: yes (drawn last)" : "Earlier: violet → blue → green → later: yellow (training rows only)";
    } else {
      categories[field].forEach((name,i)=>{const item=document.createElement("span");item.textContent=`● ${name}  `;item.style.color=palette[i%palette.length];legend.appendChild(item);});
    }
  }
  function animate(time) {
    if (!playing) return;
    if (lastTime!==null) rotate(Math.min((time-lastTime)/1000,.05)*Number(get("speed").value));
    lastTime=time;frame=requestAnimationFrame(animate);
  }
  function togglePlay() {
    playing=!playing;get("play").textContent=playing?"Ⅱ Pause":"▶ Play";lastTime=null;
    if (playing) frame=requestAnimationFrame(animate);else cancelAnimationFrame(frame);
  }
  const abort=new AbortController(), on=(element,event,fn)=>element.addEventListener(event,fn,{signal:abort.signal});
  on(get("play"),"click",togglePlay);
  on(get("angle"),"input",event=>rotate(Number(event.target.value)-angle));
  for (const role of ["first","second"]) on(get(role),"change",()=>{angle=0;get("angle").value=0;get("degrees").textContent="0°";draw();});
  on(get("reset"),"click",()=>{if(playing)togglePlay();basis=identity();angle=0;zoom=1;get("angle").value=0;get("degrees").textContent="0°";draw();});
  on(get("fit"),"click",()=>{zoom=radius/(norms.at(-1)*1.05 || 1);draw();});
  on(get("select"),"click",()=>{selectionMode=!selectionMode;get("select").setAttribute("aria-pressed",selectionMode);canvas.style.cursor=selectionMode?"crosshair":"grab";});
  on(get("clear"),"click",()=>{selected.clear();syncSelection();draw();});
  on(get("colour"),"change",draw);
  on(canvas,"keydown",event=>{if(event.key==="ArrowRight"||event.key==="ArrowLeft"){event.preventDefault();rotate(event.key==="ArrowRight"?5:-5);}if(event.code==="Space"){event.preventDefault();togglePlay();}});
  on(canvas,"wheel",event=>{event.preventDefault();zoom=Math.max(.02,Math.min(30,zoom*Math.exp(-event.deltaY*.001)));draw();});
  const point=event=>{const rect=canvas.getBoundingClientRect();return [event.clientX-rect.left,event.clientY-rect.top];};
  on(canvas,"pointerdown",event=>{if(playing)togglePlay();const [x,y]=point(event);drag={x,y,last:x,select:selectionMode||event.shiftKey};canvas.setPointerCapture(event.pointerId);tip.hidden=true;});
  on(canvas,"pointermove",event=>{
    const [x,y]=point(event);
    if(drag){if(drag.select){box={x:drag.x,y:drag.y,w:x-drag.x,h:y-drag.y};draw();}else{rotate((x-drag.last)*.35);drag.last=x;}return;}
    let nearest=-1,distance=100;
    pixels.forEach(([px,py],i)=>{const dist=(x-px)**2+(y-py)**2;if(dist<distance){distance=dist;nearest=i;}});
    tip.hidden=nearest<0;
    if(nearest>=0){const row=records[nearest];tip.textContent=`Source row ${row.source_row} · ${row.outcome}\nage ${row.age} · ${row.job}\nbalance ${row.balance} · month ${row.month}\ncampaign ${row.campaign} · previous ${row.previous}`;tip.style.left=`${Math.min(x+15,width-240)}px`;tip.style.top=`${Math.max(0,y-90)}px`;}
  });
  on(canvas,"pointerup",()=>{if(drag?.select&&box){const xmin=Math.min(box.x,box.x+box.w),xmax=Math.max(box.x,box.x+box.w),ymin=Math.min(box.y,box.y+box.h),ymax=Math.max(box.y,box.y+box.h);selected=new Set(pixels.flatMap(([x,y],i)=>x>=xmin&&x<=xmax&&y>=ymin&&y<=ymax?[i]:[]));syncSelection();}drag=null;box=null;draw();});
  on(canvas,"pointercancel",()=>{drag=null;box=null;draw();});
  on(canvas,"pointerleave",()=>{tip.hidden=true;});
  const observer=new ResizeObserver(entries=>{width=entries[0].contentRect.width;height=Math.max(360,Math.min(540,width*.58));canvas.style.height=`${height}px`;draw();});
  observer.observe(el.querySelector(".tour-stage"));draw();
  return ()=>{playing=false;cancelAnimationFrame(frame);observer.disconnect();abort.abort();};
}

export default {render};

"use strict";
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const dashboard = require('../static/js/script.js');
const tick = () => new Promise(resolve => setImmediate(resolve));
const status = () => ({schema_version:1,state:'ready',session:'session-one',
    signal:{color:'green',version:0},source:{kind:'local_video',width:720,height:480,total_frames:60},
    metrics:{processed_frames:0,frame_index:null,processing_fps:null,source_fps:30,vehicle_detections:0,
        observed_tracks:0,candidates:0,accepted_plates:0,active_viewers:1},
    recent_candidates:[],storage:{state:'disabled',evidence_capture:false},access:{history:false,evidence:false}});

class Element {
    constructor(id='') { this.id=id;this.textContent='';this.hidden=false;this.dataset={};this.children=[];
        this.attrs={};this.listeners={};this.classList={toggle(){},add(){},remove(){}}; }
    addEventListener(name,fn) {(this.listeners[name] ||= new Set()).add(fn);}
    removeEventListener(name,fn) {this.listeners[name]?.delete(fn);}
    dispatch(name,event={}) {for(const fn of this.listeners[name] || []) fn(event);}
    setAttribute(key,value) {this.attrs[key]=value;}
    removeAttribute(key) {delete this.attrs[key];}
    appendChild(child) {this.children.push(child);}
    replaceChildren(...children) {this.children=children.flatMap(c=>c.fragment?c.children:[c]);}
    get childElementCount() {return this.children.length;}
    querySelector() {return this.button;}
    showModal() {this.open=true;}
    close() {this.open=false;this.dispatch('close');}
    focus() {this.focused=true;}
}
async function fixture(handler) {
    const nodes={};
    const html=fs.readFileSync(path.join(__dirname,'../templates/index.html'),'utf8');
    for(const match of html.matchAll(/id="([^"]+)"/g)) nodes[match[1]]=new Element(match[1]);
    nodes['video-stream'].dataset.src='/video';nodes['evidence-kind'].value='crossing';
    const forms=['red','green'].map(color=>{const f=new Element();f.dataset.color=color;f.action='/set_light/'+color;f.button=new Element();return f;});
    const doc=new Element();doc.hidden=false;doc.getElementById=id=>nodes[id];doc.querySelectorAll=()=>forms;
    doc.createElement=()=>new Element();doc.createDocumentFragment=()=>{const f=new Element();f.fragment=true;return f;};
    const win=new Element();win.fetch=()=>{};
    const calls=[];let aborted=0;
    const client={request:async (...args)=>{calls.push(args);return handler?handler(...args):status();},abort(){aborted++;}};
    const app=dashboard.mount(doc,win,client);await tick();
    return {app,nodes,forms,doc,win,calls,get aborted(){return aborted;}};
}
const event=()=>({preventDefault(){}});

test('status validation rejects invalid states and signal values',()=>{
    for(const change of [{state:'fake'},{signal:{color:'yellow',version:0}},{schema_version:2},{metrics:null}])
        assert.throws(()=>dashboard.validateStatus({...status(),...change}));
});
test('invalid metrics, source shape and candidate summaries are rejected',()=>{
    for(const value of [-1,NaN,Infinity,'3',false]) {
        const data=status();data.metrics.processing_fps=value;assert.throws(()=>dashboard.validateStatus(data));
    }
    const data=status();data.recent_candidates=[{track_id:1}];assert.throws(()=>dashboard.validateStatus(data));
});
test('missing metrics are accepted as unavailable',()=>{
    const data=status();data.metrics={};assert.equal(dashboard.validateStatus(data),data);
});
test('plate validation rejects markup and invalid responses',()=>{
    assert.deepEqual(dashboard.validatePlates(['UP15AB1234']),['UP15AB1234']);
    for(const value of [{},['<script>'],['=FORMULA'],[null]]) assert.throws(()=>dashboard.validatePlates(value));
});
test('API helper rejects HTTP errors and invalid JSON',async()=>{
    await assert.rejects(dashboard.createClient(async()=>({ok:false})).request('/api/status'));
    await assert.rejects(dashboard.createClient(async()=>({ok:true,json:async()=>{throw Error('bad');}})).request('/api/status'));
});
test('API helper times out and uses no-store',async()=>{
    let cache;
    const client=dashboard.createClient((url,options)=>{cache=options.cache;return new Promise((_,reject)=>options.signal.addEventListener('abort',()=>reject(Error('timeout'))));},10);
    await assert.rejects(client.request('/api/status'));assert.equal(cache,'no-store');
});
test('API helper cancels outstanding requests',async()=>{
    const client=dashboard.createClient((url,options)=>new Promise((_,reject)=>options.signal.addEventListener('abort',()=>reject(Error('abort')))));
    const pending=client.request('/api/status');client.abort();await assert.rejects(pending);
});
test('ready state attaches exactly one stream and repeated polling reuses it',async()=>{
    const f=await fixture();try {assert.equal(f.nodes['video-stream'].src,'/video');await f.app.poll();
        assert.equal(f.nodes['video-stream'].src,'/video');assert.equal(f.calls.filter(c=>c[0]==='/violations').length,0);
    } finally {f.app.dispose();}
});
test('empty metrics render as unavailable rather than fabricated zeros',async()=>{
    const data=status();data.metrics={};const f=await fixture(()=>data);
    try {assert.equal(f.nodes['metric-detections'].textContent,'—');assert.equal(f.nodes['metric-fps'].textContent,'—');}
    finally {f.app.dispose();}
});
test('no candidates and candidates without accepted plates are distinct',async()=>{
    const data=status();data.metrics.processed_frames=10;
    const f=await fixture(()=>data);try {
        assert.match(f.nodes['candidates-empty'].textContent,/No crossing candidates/);
        data.metrics.candidates=1;await f.app.poll();assert.match(f.nodes['plates-message'].textContent,/Candidates detected/);
    } finally {f.app.dispose();}
});
test('candidate table distinguishes OCR accepted from unreadable',async()=>{
    const data=status();data.recent_candidates=[{track_id:7,frame_index:3,signal:'red',status:'candidate',recognition:'unreadable'},
        {track_id:8,frame_index:4,signal:'red',status:'candidate',recognition:'accepted'}];
    const f=await fixture(()=>data);try {
        const rows=f.nodes['candidate-rows'].children;
        assert.equal(rows[0].children[3].textContent,'Candidate');assert.equal(rows[0].children[4].textContent,'Plate unreadable');
        assert.equal(rows[1].children[4].textContent,'OCR accepted');
    } finally {f.app.dispose();}
});
test('accepted plate fetches are throttled and text-only',async()=>{
    const data=status();data.metrics.accepted_plates=1;
    const f=await fixture(url=>url==='/violations'?['UP15AB1234']:data);try {
        assert.equal(f.nodes['violations-list'].children[0].textContent,'UP15AB1234');await f.app.poll();
        assert.equal(f.calls.filter(c=>c[0]==='/violations').length,1);
    } finally {f.app.dispose();}
});
test('plate fetch failures clear stale results and report unavailable',async()=>{
    const data=status();data.metrics.accepted_plates=1;
    const f=await fixture(url=>{if(url==='/violations')throw Error('fail');return data;});try {
        assert.match(f.nodes['plates-message'].textContent,/unavailable/);assert.equal(f.nodes['violations-list'].childElementCount,0);
    } finally {f.app.dispose();}
});
test('signal update uses server response and suppresses duplicate submissions',async()=>{
    let release;let data=status();let posts=0;
    const f=await fixture((url,options)=>{
        if(options?.method==='POST'){posts++;return new Promise(resolve=>{release=resolve;});}return data;
    });try {
        const update=f.app.setSignal(event(),f.forms[0]);await tick();
        assert.equal(f.nodes['current-signal'].textContent,'GREEN');assert.ok(f.forms.every(x=>x.button.disabled));
        await f.app.setSignal(event(),f.forms[0]);assert.equal(posts,1);
        data={...status(),signal:{color:'red',version:1}};release({ok:true});await update;
        assert.equal(f.nodes['current-signal'].textContent,'RED');assert.match(f.nodes['signal-feedback'].textContent,/Server confirmed RED/);
    } finally {f.app.dispose();}
});
test('failed signal updates retain last confirmed state',async()=>{
    const f=await fixture((url,options)=>{if(options?.method==='POST')throw Error('denied');return status();});try {
        await f.app.setSignal(event(),f.forms[0]);assert.equal(f.nodes['current-signal'].textContent,'GREEN');
        assert.match(f.nodes['signal-feedback'].textContent,/Could not confirm/);
    } finally {f.app.dispose();}
});
test('older signal response cannot overwrite newer server version',async()=>{
    const f=await fixture();try {
        const fresh=status();fresh.signal={color:'red',version:2};f.app.render(fresh);f.app.render(status());
        assert.equal(f.nodes['current-signal'].textContent,'RED');
    } finally {f.app.dispose();}
});
test('polls never overlap even when a status request stalls',async()=>{
    let release;let slow=false;
    const f=await fixture(()=>slow?new Promise(resolve=>{release=resolve;}):status());try {
        slow=true;const first=f.app.poll();await f.app.poll();assert.equal(f.calls.length,2);
        release(status());await first;
    } finally {f.app.dispose();}
});
test('connection failure marks data stale and disables controls',async()=>{
    let fail=false;const f=await fixture(()=>{if(fail)throw Error('offline');return status();});try {
        fail=true;await f.app.poll();assert.equal(f.nodes['connection-message'].hidden,false);
        assert.equal(f.nodes['last-update'].textContent,'Values are stale');assert.ok(f.forms.every(x=>x.button.disabled));
    } finally {f.app.dispose();}
});
test('completion and processing error do not reconnect video',async()=>{
    const f=await fixture();try {
        for(const state of ['completed','error','stopped']) {const data=status();data.state=state;f.app.render(data);
            assert.equal(f.nodes['retry-stream'].hidden,true);assert.equal(f.nodes['video-stream'].src,'/video');}
        assert.match(f.nodes['video-message'].textContent,/stopped/);
    } finally {f.app.dispose();}
});
test('stream failure offers only an explicit reconnect',async()=>{
    const data=status();data.state='processing';const f=await fixture(()=>data);try {
        f.nodes['video-stream'].dispatch('error');assert.equal(f.nodes['retry-stream'].hidden,false);
        assert.equal(f.nodes['video-stream'].src,'/video');f.nodes['retry-stream'].dispatch('click');
        assert.equal(f.nodes['video-stream'].src,'/video?retry=1');
    } finally {f.app.dispose();}
});
test('hidden pages abort monitoring and cleanup removes listeners',async()=>{
    const f=await fixture();f.doc.hidden=true;f.doc.dispatch('visibilitychange');
    assert.ok(f.aborted>0);const before=f.calls.length;await f.app.poll();assert.equal(f.calls.length,before);
    f.app.dispose();assert.equal(f.doc.listeners.visibilitychange.size,0);
});
test('synthetic evidence dialog handles missing and failed previews',async()=>{
    const f=await fixture();try {
        f.nodes['evidence-demo'].dispatch('click');assert.equal(f.nodes['evidence-dialog'].open,true);
        assert.equal(f.nodes['evidence-image'].src,'/static/evidence-fixture.svg');
        f.nodes['evidence-kind'].value='missing';f.nodes['evidence-kind'].dispatch('change');assert.equal(f.nodes['evidence-image'].hidden,true);
        f.nodes['evidence-image'].dispatch('error');assert.match(f.nodes['evidence-caption'].textContent,/could not be loaded/);
        f.nodes['close-evidence'].dispatch('click');assert.equal(f.nodes['evidence-dialog'].open,false);
        assert.equal(f.nodes['evidence-demo'].focused,true);
    } finally {f.app.dispose();}
});

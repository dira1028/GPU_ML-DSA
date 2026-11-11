// CPU-side mathematical/source review only. Does NOT compile or execute CUDA.
// Requires Node.js (built-in crypto SHAKE is the independent output oracle).
const fs=require('fs'), path=require('path'), assert=require('assert/strict'), crypto=require('crypto');
const root=path.resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const source=read('cdp/SHA3.cu');
const array=name=>source.match(new RegExp(name+'\\[[^\\]]+\\]\\s*=\\s*\\{([^}]+)\\}'))[1].split(',').map(x=>x.trim()).filter(Boolean);
const RC=array('ROUND_CONSTANTS').map(x=>BigInt(x.replace(/ULL$/,''))), ROT=array('ROTATIONS').map(Number), PI=array('PI_INV').map(Number);
const MASK=(1n<<64n)-1n;
const rot=(a,n)=>((a<<BigInt(n))|(a>>BigInt((64-n)&63)))&MASK;
// Simulate simultaneous lane reads at each shuffle in the supplied keccak_f.
function permute(a) {
    for(const rc of RC) {
        const parity=a.map((v,t)=>{
            const x=t%5,y=Math.floor(t/5);
            for(let d=1;d<=4;++d) v^=a[x+5*((y+d)%5)];
            return v;
        });
        a=a.map((v,t)=>rot(v^parity[(t%5+4)%5]^rot(parity[(t%5+1)%5],1),ROT[t]));
        a=PI.map(i=>a[i]);
        a=a.map((v,t)=>{
            const x=t%5,base=5*Math.floor(t/5);
            return v^((~a[base+(x+1)%5])&a[base+(x+2)%5]);
        });
        a[0]^=rc;
    }
    return a;
}
class Shake {
    constructor(rate) { this.rate=rate; this.pos=0; this.a=Array(25).fill(0n); }
    absorb(buf) {
        for(const b of buf) {
            this.a[this.pos>>3]^=BigInt(b)<<BigInt(8*(this.pos&7));
            if(++this.pos===this.rate) {this.a=permute(this.a);this.pos=0;}
        }
    }
    finalize() {
        this.a[this.pos>>3]^=0x1fn<<BigInt(8*(this.pos&7));
        this.a[(this.rate-1)>>3]^=0x80n<<BigInt(8*((this.rate-1)&7));
        this.pos=this.rate;
    }
    squeeze(n) {
        const b=Buffer.alloc(n);
        for(let i=0;i<n;++i) {
            if(this.pos===this.rate) {this.a=permute(this.a);this.pos=0;}
            b[i]=Number((this.a[this.pos>>3]>>BigInt(8*(this.pos&7)))&255n); ++this.pos;
        }
        return b;
    }
}
const bytes=(n,s=0)=>Buffer.from(Array.from({length:n},(_,i)=>(i*73+s*19+11)&255));
const oracle=(rate,msg,n)=>crypto.createHash(rate===168?'shake128':'shake256',{outputLength:n}).update(msg).digest();
let checks=0;
for(const rate of [168,136]) {
    for(const len of [0,1,2,7,8,31,32,34,64,66,rate-1,rate,rate+1,2*rate,256,1088,2592]) {
        const msg=bytes(len,len), s=new Shake(rate);
        // Split absorbs, including empty and non-lane-aligned fragments.
        for(let p=0;p<len;) { const n=Math.min(1+(p*7%53),len-p);s.absorb(msg.subarray(p,p+n));p+=n; }
        s.absorb(Buffer.alloc(0));s.finalize();
        const out=Buffer.concat([s.squeeze(0),s.squeeze(1),s.squeeze(rate-1),s.squeeze(1),s.squeeze(2*rate+17)]);
        assert.deepEqual(out,oracle(rate,msg,out.length)); ++checks;
    }
}
// All 56 ExpandA mappings: rho || low(nonce) || high(nonce), unique state/output.
const rho=bytes(32), destinations=new Set();
for(let block=0;block<56;++block) {
    const r=Math.floor(block/7),c=block%7,nonce=(r<<8)+c;
    const input=Buffer.concat([rho,Buffer.from([nonce&255,nonce>>8])]);
    assert.equal(input[32],c); assert.equal(input[33],r);
    const s=new Shake(168);s.absorb(input);s.finalize();
    assert.deepEqual(Buffer.concat([s.squeeze(840),s.squeeze(168)]),oracle(168,input,1008));
    destinations.add(r*7+c);++checks;
}
assert.equal(destinations.size,56);
// ETA nonces 0..14, and gamma1 nonce wrap matches uint16_t(L*nonce+l).
for(const nonce of [...Array(15).keys(),65535,65536,7*65535+6]) {
    const v=nonce&65535,seed=bytes(64),inbuf=Buffer.concat([seed,Buffer.from([v&255,v>>8])]);
    const s=new Shake(136);s.absorb(seed);s.absorb(inbuf.subarray(64));s.finalize();
    assert.deepEqual(Buffer.concat([s.squeeze(136),s.squeeze(544)]),oracle(136,inbuf,680));++checks;
}
// Validate original sequential vs cooperative butterfly indexing for all 8 layers.
for(const inverse of [false,true]) {
    const lens=inverse?[1,2,4,8,16,32,64,128]:[128,64,32,16,8,4,2,1];
    let k=inverse?256:0;
    for(const len of lens) {
        const serial=[];
        for(let start=0;start<256;start+=2*len) {
            const z=inverse?--k:++k;
            for(let j=start;j<start+len;++j) serial.push([j,j+len,z]);
        }
        const coop=Array.from({length:128},(_,t)=>{
            const g=Math.floor(t/len),j=2*len*g+t%len;
            return [j,j+len,inverse?2*(128/len)-1-g:128/len+g];
        });
        assert.deepEqual(coop,serial);++checks;
    }
}
// Integration contracts: no old SHAKE/sampling calls in the new execution bodies.
for(const p of ['cdp/arithmetic.cuh','cdp/parent.cuh','cdp/sampling.cuh','cdp/hashes.cuh']) {
    const s=read(p).replace(/\/\/[^\n]*/g,'').replace(/\/\*[\s\S]*?\*\//g,'');
    assert(!/grid\.sync\s*\(|cudaLaunchCooperativeKernel\s*\(|cudaDeviceSynchronize\s*\(/.test(s),p);
    assert(!/PQCLEAN_MLDSA87_CLEAN_poly_(uniform|uniform_eta|uniform_gamma1|challenge)\s*\(/.test(s),p);
}
const parent=read('cdp/parent.cuh');
for(const [name,threads] of [['expand_a',25],['expand_s',25],['expand_y',25],['challenge',25],['ntt_vector',128],['sg_checks',128]]) {
    const matches=[...parent.matchAll(new RegExp(name+'<<<([^>]+)>>>','g'))];
    assert(matches.length>0);
    for(const m of matches) assert.equal(Number(m[1].split(',')[1]),threads);
}
assert(!read('cdp/SHA3.cuh').includes('steam128'));
console.log(`PASS: ${checks} mathematical/mapping checks plus source integration contracts.`);
console.log('Not validated: CUDA compilation, GPU races/runtime behavior, end-to-end ML-DSA vectors, performance.');

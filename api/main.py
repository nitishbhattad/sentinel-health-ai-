from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from api.routers import patients, chat

app = FastAPI(
    title="Healthcare Risk API",
    description="ML-powered patient risk scoring with GenAI explanations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router)
app.include_router(chat.router)

CSS = """
:root {
    --bg:#080c14; --surface:#0e1520; --card:#111827; --border:#1e2d40;
    --blue:#38bdf8; --blue2:#0ea5e9; --muted:#64748b; --text:#e2e8f0;
    --subtext:#94a3b8; --high:#f87171; --medium:#fbbf24; --low:#34d399;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
.header{background:#0e1520;border-bottom:1px solid var(--border);padding:18px 48px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100;}
.htitle{font-size:20px;font-weight:800;color:var(--blue);}
.hsub{font-size:12px;color:var(--muted);margin-top:2px;}
.hbadge{margin-left:auto;background:#0f2d1a;border:1px solid #166534;color:#4ade80;font-size:11px;padding:4px 12px;border-radius:20px;}
.container{max-width:1280px;margin:0 auto;padding:32px 48px;}
.stats-bar{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px;}
.sc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;position:relative;overflow:hidden;transition:all 0.2s;cursor:pointer;}
.sc:hover{transform:translateY(-3px);border-color:#2a3f5a;}
.sc.active{transform:translateY(-3px);}
.sc.s-total.active{border-color:var(--blue);background:#0c1f30;border-width:2px;}
.sc.s-high.active{border-color:var(--high);background:#1f0f0f;border-width:2px;}
.sc.s-medium.active{border-color:var(--medium);background:#1f1500;border-width:2px;}
.sc.s-low.active{border-color:var(--low);background:#0a1f12;border-width:2px;}
.sc.s-avg{cursor:default;}
.sc::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;}
.sc.s-total::before{background:var(--blue);}
.sc.s-high::before{background:var(--high);}
.sc.s-medium::before{background:var(--medium);}
.sc.s-low::before{background:var(--low);}
.sc.s-avg::before{background:#a78bfa;}
.slabel{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.svalue{font-size:32px;font-weight:800;letter-spacing:-1px;line-height:1;}
.sc.s-total .svalue{color:var(--blue);}
.sc.s-high .svalue{color:var(--high);}
.sc.s-medium .svalue{color:var(--medium);}
.sc.s-low .svalue{color:var(--low);}
.sc.s-avg .svalue{color:#a78bfa;}
.ssub{font-size:11px;color:var(--muted);margin-top:4px;}
.shint{font-size:10px;color:var(--muted);margin-top:6px;opacity:0.6;}
.itw{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:16px;display:none;}
.itw.show{display:block;}
.ith{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;}
.itt{font-size:13px;font-weight:700;color:var(--blue);text-transform:uppercase;letter-spacing:1px;}
.itc{background:none;border:1px solid var(--border);color:var(--muted);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;}
.itc:hover{border-color:var(--high);color:var(--high);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
.mb16{margin-bottom:16px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px;}
.ctitle{font-size:12px;font-weight:700;color:var(--blue);text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;}
input{width:100%;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:10px 14px;border-radius:8px;font-size:14px;margin-bottom:10px;outline:none;transition:border-color 0.2s;}
input:focus{border-color:var(--blue);}
.btn{background:var(--blue);color:#020d18;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-weight:700;font-size:13px;transition:all 0.2s;}
.btn:hover{background:#7dd3fc;transform:translateY(-1px);}
.btng{background:var(--surface);color:var(--text);border:1px solid var(--blue);width:100%;margin-top:8px;}
.btng:hover{background:#0c1f30;}
.result{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;margin-top:12px;font-size:13px;line-height:1.7;color:var(--subtext);min-height:60px;white-space:pre-wrap;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.5px;}
.HIGH{background:#3b0f0f;color:#f87171;border:1px solid #7f1d1d;}
.MEDIUM{background:#3b2a00;color:#fbbf24;border:1px solid #78350f;}
.LOW{background:#0a2e1a;color:#34d399;border:1px solid #14532d;}
.ICU{background:#3b0f0f;color:#f87171;border:1px solid #7f1d1d;}
.MICU{background:#3b2a00;color:#fbbf24;border:1px solid #78350f;}
.Private{background:#0a2e1a;color:#34d399;border:1px solid #14532d;}
.General{background:#0c1f3b;color:#60a5fa;border:1px solid #1e3a5f;}
.pp{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin-top:12px;display:none;}
.pp.show{display:block;}
.pph{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.ppa{width:48px;height:48px;background:linear-gradient(135deg,#0c1f30,#1e3a5f);border:2px solid var(--blue);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;}
.ppid{font-size:18px;font-weight:800;}
.pps{font-size:12px;color:var(--muted);}
.ppg{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.ppi{background:var(--card);border-radius:8px;padding:10px 12px;border:1px solid var(--border);}
.ppil{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}
.ppiv{font-size:15px;font-weight:700;margin-top:2px;}
.rbb{background:var(--border);border-radius:4px;height:6px;margin-top:4px;}
.rbf{height:6px;border-radius:4px;transition:width 0.8s ease;}
.cw{position:relative;height:200px;margin-top:8px;}
.wl{display:grid;grid-template-columns:180px 1fr;gap:16px;margin-top:12px;}
.wb{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.cm{height:260px;overflow-y:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;margin:10px 0;}
.mu{text-align:right;margin:8px 0;}
.mu span{background:var(--blue2);color:#020d18;padding:8px 14px;border-radius:16px 16px 4px 16px;display:inline-block;max-width:80%;font-size:13px;font-weight:600;}
.mb2{text-align:left;margin:8px 0;}
.mb2 span{background:var(--card);color:var(--text);padding:8px 14px;border-radius:16px 16px 16px 4px;display:inline-block;max-width:80%;font-size:13px;border:1px solid var(--border);}
.cir{display:flex;gap:8px;}
.cir input{margin:0;flex:1;}
.rt{width:100%;border-collapse:collapse;font-size:13px;}
.rt th{background:var(--surface);padding:10px 14px;text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);}
.rt td{padding:10px 14px;border-bottom:1px solid #131f2e;}
.rt tr:hover td{background:#0d1a28;cursor:pointer;}
.loading{color:var(--blue);font-size:12px;}
"""

JS = """
var API="";
var wardColors={ICU:"#f87171",MICU:"#fbbf24",Private:"#34d399",General:"#60a5fa"};
var activeFilter=null;

function riskColor(s){return s>=0.7?"#f87171":s>=0.4?"#fbbf24":"#34d399";}

function setBadge(elId,tier){
    var c={HIGH:"#f87171",MEDIUM:"#fbbf24",LOW:"#34d399"};
    var b={HIGH:"#3b0f0f",MEDIUM:"#3b2a00",LOW:"#0a2e1a"};
    var d={HIGH:"#7f1d1d",MEDIUM:"#78350f",LOW:"#14532d"};
    var el=document.getElementById(elId);
    if(el && c[tier]){
        el.innerHTML="<span style=\\"display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;background:"+b[tier]+";color:"+c[tier]+";border:1px solid "+d[tier]+";\\">"+ tier +"</span>";
    }
}

function loadStats(){
    fetch(API+"/patients/?limit=50000")
    .then(function(r){return r.json();})
    .then(function(data){
        var total=data.length;
        var high=data.filter(function(p){return p.risk_tier==="HIGH";}).length;
        var medium=data.filter(function(p){return p.risk_tier==="MEDIUM";}).length;
        var low=data.filter(function(p){return p.risk_tier==="LOW";}).length;
        var sum=data.reduce(function(s,p){return s+p.risk_score;},0);
        var avg=(sum/total*100).toFixed(1);
        document.getElementById("statTotal").textContent=total;
        document.getElementById("statHigh").textContent=high;
        document.getElementById("statMedium").textContent=medium;
        document.getElementById("statLow").textContent=low;
        document.getElementById("statAvg").textContent=avg+"%";
        buildChart(data);
    })
    .catch(function(e){console.error("Stats error:",e);});
}

function buildChart(data){
    var buckets=[0,0,0,0,0,0,0,0,0,0];
    data.forEach(function(p){
        var i=Math.min(Math.floor(p.risk_score*10),9);
        buckets[i]++;
    });
    var labels=["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"];
    var colors=buckets.map(function(_,i){return i>=7?"#f87171":i>=4?"#fbbf24":"#34d399";});
    new Chart(document.getElementById("riskChart").getContext("2d"),{
        type:"bar",
        data:{labels:labels,datasets:[{label:"Patients",data:buckets,backgroundColor:colors,borderRadius:6,borderSkipped:false}]},
        options:{
            responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false}},
            scales:{
                x:{ticks:{color:"#64748b",font:{size:10}},grid:{color:"#1e2d40"}},
                y:{ticks:{color:"#64748b",font:{size:10}},grid:{color:"#1e2d40"}}
            }
        }
    });
}

function filterPatients(tier){
    if(activeFilter===tier){closeTable();return;}
    activeFilter=tier;
    ["ALL","HIGH","MEDIUM","LOW"].forEach(function(t){
        var el=document.getElementById("card-"+t);
        if(el)el.classList.remove("active");
    });
    var ac=document.getElementById("card-"+tier);
    if(ac)ac.classList.add("active");
    var wrap=document.getElementById("inlineTableWrap");
    wrap.classList.add("show");
    document.getElementById("inlineTableContainer").innerHTML="<span class=\\"loading\\">Loading...</span>";
    var titles={ALL:"All Patients",HIGH:"High Risk Patients",MEDIUM:"Medium Risk Patients",LOW:"Low Risk Patients"};
    document.getElementById("inlineTableTitle").textContent=titles[tier];
    setTimeout(function(){wrap.scrollIntoView({behavior:"smooth",block:"start"});},100);
    var url=tier==="ALL"?API+"/patients/?limit=1000":API+"/patients/?tier="+tier+"&limit=500";
    fetch(url)
    .then(function(r){return r.json();})
    .then(function(data){
        if(!data.length){
            document.getElementById("inlineTableContainer").innerHTML="<p>No patients found.</p>";
            return;
        }
        var rows=data.map(function(p){
            var rc=riskColor(p.risk_score);
            var bc={HIGH:"#3b0f0f",MEDIUM:"#3b2a00",LOW:"#0a2e1a"};
            var tc={HIGH:"#f87171",MEDIUM:"#fbbf24",LOW:"#34d399"};
            var bd={HIGH:"#7f1d1d",MEDIUM:"#78350f",LOW:"#14532d"};
            var bh="<span style=\\"display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;background:"+bc[p.risk_tier]+";color:"+tc[p.risk_tier]+";border:1px solid "+bd[p.risk_tier]+";\\">"+ p.risk_tier +"</span>";
            return "<tr onclick=\\"selectPatient("+p.subject_id+")\\">"
                +"<td style=\\"color:var(--text);font-weight:700;\\">"+p.subject_id+"</td>"
                +"<td style=\\"color:"+rc+";font-weight:700;\\">"+(p.risk_score*100).toFixed(1)+"%</td>"
                +"<td>"+bh+"</td>"
                +"<td>"+p.admission_count+"</td>"
                +"<td>"+(p.emergency_ratio*100).toFixed(1)+"%</td>"
                +"</tr>";
        }).join("");
        document.getElementById("inlineTableContainer").innerHTML=
            "<p style=\\"font-size:12px;color:var(--muted);margin-bottom:12px;\\">"+data.length+" patients</p>"
            +"<table class=\\"rt\\"><thead><tr><th>Patient ID</th><th>Risk Score</th><th>Tier</th><th>Admissions</th><th>Emergency %</th></tr></thead>"
            +"<tbody>"+rows+"</tbody></table>";
    })
    .catch(function(e){
        document.getElementById("inlineTableContainer").innerHTML="<span>Error: "+e.message+"</span>";
    });
}

function closeTable(){
    document.getElementById("inlineTableWrap").classList.remove("show");
    ["ALL","HIGH","MEDIUM","LOW"].forEach(function(t){
        var el=document.getElementById("card-"+t);
        if(el)el.classList.remove("active");
    });
    activeFilter=null;
}

function selectPatient(id){
    document.getElementById("searchId").value=id;
    searchPatient();
    document.querySelector(".grid2").scrollIntoView({behavior:"smooth"});
}

function searchPatient(){
    var id=document.getElementById("searchId").value;
    if(!id)return;
    document.getElementById("searchError").style.display="none";
    document.getElementById("patientProfile").classList.remove("show");
    fetch(API+"/patients/"+id+"/risk")
    .then(function(r){return r.json();})
    .then(function(d){
        var rc=riskColor(d.risk_score);
        var wc=wardColors[d.predicted_ward]||"#60a5fa";
        document.getElementById("profileAvatar").textContent=d.risk_tier==="HIGH"?"!":d.risk_tier==="MEDIUM"?"~":"OK";
        document.getElementById("profileId").textContent="Patient "+d.subject_id;
        document.getElementById("profileSub").textContent=d.risk_tier+" RISK - Ward: "+(d.predicted_ward||"N/A");
        document.getElementById("profileRisk").style.color=rc;
        document.getElementById("profileRisk").textContent=(d.risk_score*100).toFixed(1)+"%";
        setBadge("profileTier",d.risk_tier);
        document.getElementById("profileWard").style.color=wc;
        document.getElementById("profileWard").textContent=d.predicted_ward||"N/A";
        document.getElementById("profileAdmissions").textContent=d.admission_count;
        document.getElementById("profileEmergency").textContent=(d.emergency_ratio*100).toFixed(1)+"%";
        document.getElementById("profileStay").textContent=(d.avg_los_days||0).toFixed(1)+" days";
        document.getElementById("profileRiskBar").style.width=(d.risk_score*100)+"%";
        document.getElementById("profileRiskBar").style.background=rc;
        document.getElementById("patientProfile").classList.add("show");
        ["riskPatientId","explainPatientId","wardPatientId","chatPatientId"].forEach(function(x){
            document.getElementById(x).value=id;
        });
    })
    .catch(function(e){
        document.getElementById("searchError").textContent="Patient "+id+" not found";
        document.getElementById("searchError").style.display="block";
    });
}

function getRisk(){
    var id=document.getElementById("riskPatientId").value;
    if(!id)return;
    document.getElementById("riskResult").innerHTML="<span class=\\"loading\\">Loading...</span>";
    fetch(API+"/patients/"+id+"/risk")
    .then(function(r){return r.json();})
    .then(function(d){
        document.getElementById("riskResult").textContent="Patient "+d.subject_id+" | Risk: "+(d.risk_score*100).toFixed(1)+"% ["+d.risk_tier+"] | Admissions: "+d.admission_count+" | Emergency: "+(d.emergency_ratio*100).toFixed(1)+"% | Avg Stay: "+(d.avg_los_days||0).toFixed(1)+" days";
    })
    .catch(function(e){document.getElementById("riskResult").textContent="Error: "+e.message;});
}

function downloadReport(){
    var id=document.getElementById("riskPatientId").value;
    if(!id){alert("Enter a patient ID first");return;}
    var btn=document.getElementById("reportBtn");
    btn.textContent="Generating PDF...";btn.disabled=true;
    fetch(API+"/patients/"+id+"/report")
    .then(function(r){
        if(!r.ok)throw new Error("Failed");
        return r.blob();
    })
    .then(function(blob){
        var url=URL.createObjectURL(blob);
        var a=document.createElement("a");
        a.href=url;a.download="patient_"+id+"_report.pdf";a.click();
        URL.revokeObjectURL(url);
        btn.textContent="Downloaded!";btn.disabled=false;
        setTimeout(function(){btn.textContent="Download PDF Report";},3000);
    })
    .catch(function(e){btn.textContent="Error: "+e.message;btn.disabled=false;});
}

function getExplanation(){
    var id=document.getElementById("explainPatientId").value;
    if(!id)return;
    document.getElementById("explainResult").innerHTML="<span class=\\"loading\\">Generating...</span>";
    fetch(API+"/patients/"+id+"/explain")
    .then(function(r){return r.json();})
    .then(function(d){
        document.getElementById("explainResult").textContent=d.risk_tier+" "+(d.risk_score*100).toFixed(1)+"% - "+d.explanation;
    })
    .catch(function(e){document.getElementById("explainResult").textContent="Error: "+e.message;});
}

function getWardTimeline(){
    var id=document.getElementById("wardPatientId").value;
    if(!id)return;
    document.getElementById("wardBadge").innerHTML="<div>...</div>";
    document.getElementById("wardTimeline").innerHTML="<span class=\\"loading\\">Generating discharge plan...</span>";
    fetch(API+"/patients/"+id+"/ward")
    .then(function(r){return r.json();})
    .then(function(d){
        var color=wardColors[d.predicted_ward]||"#60a5fa";
        document.getElementById("wardBadge").innerHTML=
            "<div style=\\"font-size:32px;font-weight:800;color:"+color+";\\">"+ d.predicted_ward +"</div>"
            +"<div style=\\"font-size:11px;margin-top:6px;color:var(--muted);\\">~"+(d.estimated_los_days||0).toFixed(1)+" days</div>"
            +"<div style=\\"margin-top:8px;font-size:11px;font-weight:700;color:"+color+";\\">"+ d.risk_tier +"</div>"
            +"<div style=\\"font-size:11px;margin-top:4px;color:"+color+";\\">"+(d.risk_score*100).toFixed(1)+"% risk</div>";
        document.getElementById("wardTimeline").textContent=d.discharge_timeline;
    })
    .catch(function(e){
        document.getElementById("wardBadge").innerHTML="<div>Error</div>";
        document.getElementById("wardTimeline").textContent="Error: "+e.message;
    });
}

function sendChat(){
    var id=document.getElementById("chatPatientId").value;
    var question=document.getElementById("chatInput").value.trim();
    if(!id||!question)return;
    var messages=document.getElementById("chatMessages");
    messages.innerHTML+="<div class=\\"mu\\"><span>"+question+"</span></div>";
    messages.innerHTML+="<div class=\\"mb2\\" id=\\"typing\\"><span class=\\"loading\\">Thinking...</span></div>";
    document.getElementById("chatInput").value="";
    messages.scrollTop=messages.scrollHeight;
    fetch(API+"/chat/",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({subject_id:parseInt(id),question:question})
    })
    .then(function(r){return r.json();})
    .then(function(d){
        var t=document.getElementById("typing");
        if(t)t.outerHTML="<div class=\\"mb2\\"><span>"+d.answer+"</span></div>";
        messages.scrollTop=messages.scrollHeight;
    })
    .catch(function(e){
        var t=document.getElementById("typing");
        if(t)t.outerHTML="<div class=\\"mb2\\"><span>Error: "+e.message+"</span></div>";
    });
}

loadStats();
"""

HTML = """<!DOCTYPE html>
<html>
<head>
<title>Healthcare Risk Platform</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div class="header">
    <div style="font-size:28px;">&#x1F3E5;</div>
    <div>
        <div class="htitle">Healthcare Risk Platform</div>
        <div class="hsub">ML &middot; SHAP &middot; RAG &middot; LLM &middot; Docker</div>
    </div>
    <div class="hbadge">&#x25CF; LIVE</div>
</div>
<div class="container">

<div class="stats-bar">
    <div class="sc s-total" onclick="filterPatients('ALL')" id="card-ALL">
        <div class="slabel">Total Patients</div>
        <div class="svalue" id="statTotal">-</div>
        <div class="ssub">in database</div>
        <div class="shint">click to view all</div>
    </div>
    <div class="sc s-high" onclick="filterPatients('HIGH')" id="card-HIGH">
        <div class="slabel">High Risk</div>
        <div class="svalue" id="statHigh">-</div>
        <div class="ssub">need attention</div>
        <div class="shint">click to view</div>
    </div>
    <div class="sc s-medium" onclick="filterPatients('MEDIUM')" id="card-MEDIUM">
        <div class="slabel">Medium Risk</div>
        <div class="svalue" id="statMedium">-</div>
        <div class="ssub">monitor closely</div>
        <div class="shint">click to view</div>
    </div>
    <div class="sc s-low" onclick="filterPatients('LOW')" id="card-LOW">
        <div class="slabel">Low Risk</div>
        <div class="svalue" id="statLow">-</div>
        <div class="ssub">stable patients</div>
        <div class="shint">click to view</div>
    </div>
    <div class="sc s-avg">
        <div class="slabel">Avg Risk Score</div>
        <div class="svalue" id="statAvg">-</div>
        <div class="ssub">platform average</div>
    </div>
</div>

<div class="itw" id="inlineTableWrap">
    <div class="ith">
        <div class="itt" id="inlineTableTitle">Patients</div>
        <button class="itc" onclick="closeTable()">x Close</button>
    </div>
    <div id="inlineTableContainer"><span class="loading">Loading...</span></div>
</div>

<div class="grid2">
    <div class="card">
        <div class="ctitle">Patient Search</div>
        <div style="display:flex;gap:8px;">
            <input id="searchId" type="number" placeholder="Enter Patient ID..." style="margin:0;flex:1;" onkeypress="if(event.key===&apos;Enter&apos;)searchPatient()"/>
            <button class="btn" onclick="searchPatient()">Search</button>
        </div>
        <div class="pp" id="patientProfile">
            <div class="pph">
                <div class="ppa" id="profileAvatar">&#x1F464;</div>
                <div>
                    <div class="ppid" id="profileId">Patient -</div>
                    <div class="pps" id="profileSub">-</div>
                </div>
            </div>
            <div class="ppg">
                <div class="ppi"><div class="ppil">Risk Score</div><div class="ppiv" id="profileRisk">-</div></div>
                <div class="ppi"><div class="ppil">Risk Tier</div><div class="ppiv" id="profileTier">-</div></div>
                <div class="ppi"><div class="ppil">Ward</div><div class="ppiv" id="profileWard">-</div></div>
                <div class="ppi"><div class="ppil">Admissions</div><div class="ppiv" id="profileAdmissions">-</div></div>
                <div class="ppi"><div class="ppil">Emergency Rate</div><div class="ppiv" id="profileEmergency">-</div></div>
                <div class="ppi"><div class="ppil">Avg Stay</div><div class="ppiv" id="profileStay">-</div></div>
            </div>
            <div style="margin-top:10px;">
                <div style="font-size:10px;color:var(--muted);">RISK LEVEL</div>
                <div class="rbb"><div class="rbf" id="profileRiskBar" style="width:0%;background:#f87171;"></div></div>
            </div>
        </div>
        <div id="searchError" style="display:none;color:#f87171;font-size:12px;margin-top:8px;"></div>
    </div>
    <div class="card">
        <div class="ctitle">Risk Score Distribution</div>
        <div class="cw"><canvas id="riskChart"></canvas></div>
    </div>
</div>

<div class="grid2">
    <div class="card">
        <div class="ctitle">Patient Risk Score</div>
        <input id="riskPatientId" type="number" placeholder="Patient ID (e.g. 284)"/>
        <button class="btn" onclick="getRisk()">Get Risk Score</button>
        <div class="result" id="riskResult">Enter a patient ID above...</div>
        <button class="btn btng" onclick="downloadReport()" id="reportBtn">Download PDF Report</button>
    </div>
    <div class="card">
        <div class="ctitle">AI Risk Explanation</div>
        <input id="explainPatientId" type="number" placeholder="Patient ID (e.g. 284)"/>
        <button class="btn" onclick="getExplanation()">Generate Explanation</button>
        <div class="result" id="explainResult">Enter a patient ID above...</div>
    </div>
</div>

<div class="card mb16">
    <div class="ctitle">Ward Assignment and Discharge Timeline</div>
    <div style="display:flex;gap:8px;margin-bottom:4px;">
        <input id="wardPatientId" type="number" placeholder="Patient ID (e.g. 284)" style="width:280px;margin:0;"/>
        <button class="btn" onclick="getWardTimeline()">Get Ward and Discharge Plan</button>
    </div>
    <div class="wl">
        <div class="wb" id="wardBadge">
            <div style="font-size:36px;">&#x1F3E5;</div>
            <div style="font-size:13px;color:var(--muted);margin-top:8px;">Enter patient ID</div>
        </div>
        <div class="result" id="wardTimeline">Enter a patient ID to see ward assignment and discharge plan...</div>
    </div>
</div>

<div class="card mb16">
    <div class="ctitle">Clinical RAG Chatbot</div>
    <input id="chatPatientId" type="number" placeholder="Patient ID" style="width:200px;display:inline-block;margin-right:8px;margin-bottom:8px;"/>
    <span style="font-size:12px;color:var(--muted);">set patient context</span>
    <div class="cm" id="chatMessages">
        <div class="mb2"><span>Enter a patient ID and ask anything about their clinical history.</span></div>
    </div>
    <div class="cir">
        <input id="chatInput" type="text" placeholder="Ask about symptoms, medications, admissions..." onkeypress="if(event.key===&apos;Enter&apos;)sendChat()"/>
        <button class="btn" onclick="sendChat()">Send</button>
    </div>
</div>

</div>
<script>{JS}</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML.replace("{CSS}", CSS).replace("{JS}", JS)


@app.get("/health")
def health():
    return {"status": "ok", "service": "Healthcare Risk API"}
"""Appendix figure: r(ΔLLM, ΔBToM) Every-step vs Prefix-step.
Usage: python analysis/plot_update_every_prefix.py <dir containing stats/update/update_agreement.csv> <out_dir>"""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S,O=sys.argv[1],sys.argv[2]
M=[("Claude Opus 4.6","Claude\nOpus"),("Gemini 2.5 Pro","Gemini\nPro"),("GPT-5.4","GPT\n5.4")]
U=pd.read_csv(f"{S}/stats/update/update_agreement.csv"); U=U[(U.reference=="BToM")&(U.phase_idx.astype(str)=="all")]
fig,axs=plt.subplots(1,2,figsize=(3.6,2.1),sharey=True,gridspec_kw=dict(wspace=0.12))
for ax,dim in zip(axs,["belief","desire"]):
    for i,(um,_) in enumerate(M):
        for c,off in [("Every-step",-0.19),("Prefix-step",0.19)]:
            r=U[(U.model==um)&(U.dimension==dim)&(U.condition==c)].iloc[0]
            kw=dict(facecolor="white",edgecolor="#888780",hatch="////") if c=="Every-step" else dict(facecolor="#1D9E75",edgecolor="#1D9E75")
            ax.bar(i+off,r.r,width=0.34,lw=0.8,label=c if i==0 else None,**kw)
            ax.errorbar(i+off,r.r,[[r.r-r.ci_low],[r.ci_high-r.r]],color="#333",lw=0.8,capsize=1.8)
    ax.axhline(0,color="#333",lw=0.6); ax.set_title(dim.capitalize(),fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([t for _,t in M],fontsize=6.5)
    ax.tick_params(axis="y",labelsize=6.5,length=2); ax.tick_params(axis="x",length=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y",color="#e6e6e6",lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("r(ΔLLM, ΔBToM)",fontsize=7)
fig.legend(*axs[0].get_legend_handles_labels(),loc="upper center",ncol=2,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,1.08))
fig.savefig(f"{O}/figS_update_every_prefix.png",dpi=300,bbox_inches="tight")

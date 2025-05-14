/* ========================================================================
   Modern FAQ Chat Widget  –  https://yourdomain.com/widget.js
   ======================================================================== */

   (function () {
    /* ---------- 1. CONFIG ------------------------------------------------ */
    const TAG          = document.currentScript;
    const API_URL      = TAG.getAttribute("data-api") || "https://your-api-domain.com/ask";
    const THEME_COLOR  = TAG.getAttribute("data-color") || "#4f46e5";
    const GREETING     = TAG.getAttribute("data-greeting") || "Need help? Ask away!";
  
    /* ---------- 2. STYLES ------------------------------------------------ */
    const style = document.createElement("style");
    style.textContent = `
      :root{
        --faq-theme: ${THEME_COLOR};
        --bot-bg: #f1f5f9;
        --user-bg: var(--faq-theme);
      }
      #faq-btn{
        position:fixed;bottom:20px;right:20px;width:58px;height:58px;
        border:none;border-radius:50%;background:var(--faq-theme);color:#fff;
        font-size:28px;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,.25);
        transition:background .2s;z-index:9999
      }
      #faq-btn:hover{background:#3b39d1}
      #faq-box{
        position:fixed;bottom:90px;right:20px;display:none;width:330px;height:460px;
        border-radius:16px;background:#fff;box-shadow:0 6px 24px rgba(0,0,0,.2);
        font-family:Inter,Arial,sans-serif;z-index:9999;flex-direction:column;
        overflow:hidden;animation:pop .25s ease-out forwards
      }
      @keyframes pop{from{opacity:0;transform:scale(.8)}to{opacity:1;transform:scale(1)}}
      #faq-head{background:var(--faq-theme);color:#fff;padding:12px 16px;font-size:16px;font-weight:600}
      #faq-msgs{flex:1;padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
      .msg{max-width:80%;padding:10px 14px;border-radius:18px;font-size:14px;line-height:1.4;opacity:0;animation:fade .3s forwards}
      .bot{background:var(--bot-bg);color:#111;border-bottom-left-radius:4px}
      .user{background:var(--user-bg);color:#fff;margin-left:auto;border-bottom-right-radius:4px}
      @keyframes fade{to{opacity:1}}
      #faq-input{display:flex;border-top:1px solid #eee}
      #faq-input input{flex:1;border:none;padding:14px;font-size:14px;outline:none}
      #faq-input button{border:none;background:var(--faq-theme);color:#fff;padding:0 18px;font-weight:600;cursor:pointer}
    `;
    document.head.appendChild(style);
  
    /* ---------- 3. DOM --------------------------------------------------- */
    const btn = document.createElement("button");
    btn.id = "faq-btn";
    btn.textContent = "💬";
    document.body.appendChild(btn);
  
    const box = document.createElement("div");
    box.id = "faq-box";
    box.innerHTML = `
      <div id="faq-head">${GREETING}</div>
      <div id="faq-msgs"></div>
      <div id="faq-input">
        <input type="text" placeholder="Type your question…">
        <button>Send</button>
      </div>`;
    document.body.appendChild(box);
  
    /* ---------- 4. Helpers ---------------------------------------------- */
    const msgs    = box.querySelector("#faq-msgs");
    const input   = box.querySelector("input");
    const sendBtn = box.querySelector("button");
  
    function bubble(role, text){
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.textContent = text;
      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
    }
  
    /* -- typing indicator -- */
    let typingBubble = null;
    function showTyping(){
      if(typingBubble) return;
      typingBubble = document.createElement("div");
      typingBubble.className = "msg bot";
      typingBubble.style.fontStyle = "italic";
      typingBubble.textContent = "Bot is typing…";
      msgs.appendChild(typingBubble);
      msgs.scrollTop = msgs.scrollHeight;
    }
    function replaceTyping(answer){
      if(!typingBubble) return;
      typingBubble.style.fontStyle = "normal";
      typingBubble.textContent = answer;
      typingBubble = null;
    }
  
    async function ask(){
      const q = input.value.trim();
      if(!q) return;
      bubble("user", q);
      input.value = "";
      showTyping();
      try{
        const res = await fetch(API_URL,{
          method:"POST",
          headers:{ "Content-Type":"application/json" },
          body:JSON.stringify({ question:q })
        });
        const data = await res.json();
        replaceTyping(data.answer || "Sorry, I’m not sure.");
      }catch(e){
        replaceTyping("Error contacting server.");
      }
    }
  
    /* ---------- 5. Events ------------------------------------------------ */
    btn.onclick = () => {
      const open = box.style.display === "flex";
      box.style.display = open ? "none" : "flex";
      if(!open) input.focus();
    };
    sendBtn.onclick = ask;
    input.addEventListener("keypress",e=>{
      if(e.key==="Enter"){e.preventDefault();ask();}
    });
  })();
  
# AI Assistance for Scala Developers: What JetBrains Brings to the Table

### Abstract

In the last few years, one of the leading new areas of development in JetBrains has been AI assistance. It started with simple access to an external large language model within IntelliJ IDEA—but we didn’t stop there.

Today, JetBrains offers a wide range of new features and extensions that can make software developers work more quickly and productively. During this talk, we will go through the most important of them - AI Assistant, local and cloud completion, connecting to local LLMs, the integration of Model Context Protocol, and more. We will discuss how these tools can boost your productivity and improve your coding experience. We will also delve into… sorry, I mean, we will talk about JetBrains Junie, AI agents in general, and the future of AI assistance.

The talk will consist of animated infographics and live coding examples.

### Topics
* AI Assistant
  * AI Chat window
    * Including codebase (the current file, more files)
    * Add attachments
    * Choose LLM (on the server)
    * Local LLMs: LM Studio, Ollama (maybe show an example of how it works)
      * Local LLMs work only in AI Chat, not in code completions
    * Chat vs Edit mode
    * /commands and #mentions
    * AI Chat settings popup
      * Rename
      * Disable
      * Learn features
      * Mention that we will talk about the license at the end of the talk
  * Code Completion
    * https://youtrack.jetbrains.com/projects/JBAI/articles/JBAI-A-6/AI-Code-Completion
    * Mention traditional code completion (type inference, fields and methods lookup)
    * Local completions
    * Cloud-based completions
      * Multi-line completions
    * Names suggestions
    * Sorting hints according to AI Assistant judgement
    * A shortcut for AI code completions
    * Completion policy (focused, creative)
    * Popup suggestions vs inline suggestions
  * Generate code from prompt
    * Prompt input box in editor
    * Suggest refactorings
    * Try to find problems and suggest fixes
  * Generate commit messages
    * Explain the VCS situation
    * Explain commits
  * Explain code
  * Generate tests
  * Generate documentation
  * Explain errors in the runtime popup window
  * Custom AI commands
  * Next Edit feature (still in-progress)
* Model Context Protocol
  * https://en.wikipedia.org/wiki/Model_Context_Protocol
  * https://modelcontextprotocol.io/specification/2025-03-26
* Mellum
  * https://github.com/mellumai/mellum
  * 4B parameters, focused solely on code completion
  * Open-source on HuggingFace
* Junie
  * How to install Junie from AI Assistant (currently unavailable for IJ 2025.2 EAP)
  * https://jetbrains.com/junie
  * What LLM is used in Junie?
  * Ask mode: like AI Assistant chat
  * Brave mode: allows Junie to execute commands without asking for confirmation
  * How to use Junie?
    * https://www.youtube.com/watch?v=fcbSG8lm7So
    * Give it a programming task, let it create new code files
    * Give it an execution task, let it run the program and/or run commands in the terminal
* Licensing
  * TBD
* What LLM can't help you with
  * A short explanation how LLM works
  * No creative work, poor logical/mathematical reasoning
  * Review what LLM does



### Notes

* An example of command completion?
* Focus on Next Edit a bit more than just talking about the old features of AI Assistant
* There is an experiment of using Claude in AI Assistant as an AI Agent
* Maybe explain what is an AI Agent in general?
* Junie work slow - it's better not to do live coding.
  * So the talk would be slides. Explain at the beginning that this is because there's so much to show.

* I can make a video recording and show it during the talk with explanation
* Make a video recording for AI Assistant as well, just as a fallback
* Check with advocates and AI Team if we have a Q&A document we can use for questions from the audience
  * If there's no Q&A doc, write one.

* Watch JetBrains videos about AI Assistant, Junie, and Mellum
* Ask AI Team about MCP.
* "Why is there AI Assistant and Junie?"
* "What's up with quotas? -- licensing?"
* Check with IntelliJ IDEA marketing if a video about using Ai Assistant or Junie for Scala coding is something they would be interested in releasing as the AI Assistance promotion.
* We can make reels of how Junie helps with Scala
* Example for Junie: Fixing an issue in mUnit? Or something from Li Haoyi?
* Junie works the best if you describe carefully what you want. Look for the talk by Baruch from the IJ conference, about comparing Junie and Windsurf.

### Questions
I'm working on a conference talk about AI assistance in IntelliJ IDEA, with focus on how it can help Scala developers. I went through tutorials on AI Assistant and Junie, experimented with both, made my own code example that I would like to show at the talk, etc. 
Below you will find a list of questions I still have. Can you help me with them? 
* Why are there both AI Assistant and Junie? 
  * I have my own idea how to answer but I'd like to know some other takes.
* Licensing and quotas
  * If I understand correctly, there were some changes, and the topic is sensitive, so I prefer to ask you about even basic things, so that I won't make a mistake.
  * I need a source with all the info about licenses that I can check just before my talk. A website or a document. Is there something like that?
  * If someone uses IntelliJ IDEA Community Edition, how does their access to AI Assistant and Junie looks like?
  * Is it different for someone who uses Ultimate but without any additional licensing?
  * What are the additional licenses for AI Assistant and Junie?
  * Are there tokens limits for AI Assistant and Junie? Do they vary depending on the license?
  * What happens when the user exceeds the token limit? Can they buy more? How?
* LLM in Junie
  * I know Claude 3.7 and 4.0 are available. Are there plans for more? 
  * Why is Claude 3.7 the default?
  * Is their usage affected by licensing?
* How to use Mellum in AI Assistant and Junie? 
  * If it's not possible yet, then why and when will it be?
  * What are the future plans for Mellum?
  * Is it possible to fine-tune it for Scala? If yes, how?
* Would you be interested in a video about using AI Assistant or Junie for coding in Scala?
* MCP: From what I read, it looks like it's not a subject that connects well with the rest of my talk. Is there something special about MCP as used in JetBrains that you would like me to mention? If not, I would like to drop it.

---

Przeprowadzka 500kg, 5.3m3 (JetBrains da mi kontakt do firmy przeprowadzkowej), tym zajmuje się Kerstin Feuerstein.

Bilety x2 TravelPerk

Badania medyczne w Medicover w Warszawie

Spotkanie w biurze, 4 sierpnia (początek pracy 1 sierpnia) -- być w Warszawie! Szkolenie BHP będzie!

---

https://www.youtube.com/watch?v=oUGIcp2Iph0&list=WL&index=1
Junie Livestream #1: Meet Junie, the Coding Agent by JetBrains | Live Demo + Tips from Anton Arhipov


https://www.youtube.com/watch?v=hUXGR0a9NLs&t=13s
Interview with JBaruch
Notes:
* Main problem with vibe coding is self-verification. If AI made a mistake and you want it to correct it, there's nothing stopping AI from making a mistake in correction.
* 



---

Praktyczność, Zdrowie / kondycja, Fajność, Community -- PZFC, 3 - bardzo, 2 - średnio, 1 - mało

Kickboxing 3+2+3+1 = 9

Karate ashihara 2+3+3+3 = 11

Karate oyama (kyokushin) 2+3+2+3 = 10

Sambo 3+3+1+1 = 8

Aikido 1+2+2+3 = 8

Muay Thai 3+3+3+1 = 10

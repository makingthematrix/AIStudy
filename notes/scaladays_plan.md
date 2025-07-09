# AI Assistance for Scala Developers
## What JetBrains Brings to the Table


# Abstract

In the last few years, one of the leading new areas of development in JetBrains has been AI assistance. It started with simple access to an external large language model within IntelliJ IDEA—but we didn’t stop there.

Today, JetBrains offers a wide range of new features and extensions that can make software developers work more quickly and productively. During this talk, we will go through the most important of them – AI Assistant, local and cloud completion, connecting to local LLMs, the integration of Model Context Protocol, and more. We will discuss how these tools can boost your productivity and improve your coding experience. We will also delve into… sorry, I mean, we will talk about JetBrains Junie, AI agents in general, and the future of AI assistance.

The talk will consist of animated infographics and live coding examples.


## Notes about the abstract



* Due to time constraints and a loose connection to the rest of the talk, I will probably drop MCP.
* On the other hand, Junie gets more screen time – literally, there will be a video.
* Instead of animated infographics, I have recently preferred to make many slides that differ only by one small element. By clicking through them quickly, I make an impression of "animation." It's easier to control and takes less time to show, as I don't have to switch from slides to a video app.
* I will update the abstract and send it to the organizers.


# Plan



* The slides with the agenda
    * "Hello from JetBrains", we are a Gold Sponsor, who am I
    * Topics we will go through: Code completion, AI Assistant, Mellum, Junie, licensing, and a few tips on how to use AI in coding
* The code example we will work on: a binary tree
    * A quick explanation of what it is and why I show it to the audience
    * Throughout the talk, I will switch between live coding and slides that explain what's happening in the code.
    * The binary tree code is already in a state where I can create a tree, add new elements to it, and print it out to the terminal.
    * GitHub repository: [https://github.com/makingthematrix/binarytree](https://github.com/makingthematrix/binarytree)
* Code completion
    * Mention traditional code completion (type inference, fields, and methods lookup)
    * Open Settings and go through the features under Code Completion and Inline Completion
        * Local completions
        * Cloud-based completions
        * Multi-line completions
        * A shortcut for AI code completions
        * Names suggestions
        * Sorting hints according to the AI Assistant's judgment
        * Completion policy (focused, creative)
        * Pop-up suggestions vs inline suggestions
    * **Note**: A lot of those features are called "... Completions". Be careful to explain the differences correctly. [(This might help)](https://youtrack.jetbrains.com/projects/JBAI/articles/JBAI-A-6/AI-Code-Completion)
    * **Code example #1:** Write the "add" method for simultaneously adding many elements to the tree. Show inline code completion, name suggestions, and the in-editor prompt. Possibly write the same method twice using different features (it's a one-liner)
    * **Code example #2: **Write the "size" method, which recursively counts the number of elements in the tree. For now, use traditional code completion, e.g., match-exhaustive. We will come back to it.
    * Mention the **Next Edit** feature, show a short example (Next Edit is still in progress, let's come back to this point in August)
* Mellum
    * Mellum is already used for code completion in IntelliJ IDEA.
    * Show [https://github.com/mellumai/mellum](https://github.com/mellumai/mellum)
    * Describe the main characteristics, and mention that Mellum is open source.
    * Show [its page on HuggingFace](https://huggingface.co/JetBrains/Mellum-4b-base).
    * Future plans for Mellum.
    * The possibility of fine-tuning Mellum for Scala.
* AI Assistant
    * How to install from Settings | Plugins.
    * Open the Open AI Chat window and go through the features:
    * How to include attachments.
    * How to manage different chats (create, rename, delete).
    * How to choose between models.
    * **Show a short video** on connecting to a local model through ollama.
    * Chat vs Edit mode: In the chat mode, let the AI Assistant explain the code in Tree.scala.
    * **Code example #3**: Rewrite the "size" method with the help of the AI Assistant in the Edit mode.
    * Make the AI Assistant generate a git commit message for the new code. (I might skip it due to time constraints)
    * How to learn more: Show the "Discover Features" panel.
    * Mention licensing, and that we will talk about it later.
* Junie
    * How to install Junie from AI Assistant (currently unavailable for IJ 2025.2 EAP)
    * [Mention the webpage ](https://jetbrains.com/junie)– Junie is in development, so it's important to check the page and the blog to learn about new features and changes.
    * Mention what LLM is used in Junie (Claude 3.7 and 4.0).
    * Mention Ask mode: works like the AI Assistant chat.
    * Why are there both AI Assistant and Junie? (Note that the Edit mode in AI Assistant)
    * Mention Brave mode: allows Junie to execute commands without asking for confirmation.
    * How to use Junie?
        * **Show a video** with Junie writing the code for iterating over the tree.
        * Show how the iterator should work (on the slides) and what is difficult about it.
        * The first solution is imperative and messy.
        * The second solution is Scala-idiomatic, but it's not optimal due to how the prompt was phrased: Junie coded the "toList" method internally and used it only to return its iterator.
* A few words on AI agents and LLM in general
    * Slides + a short discussion of limitations inherent to LMMs
    * Hints:
        * Be careful about prompts.
        * Check the results often.
        * Don't let AI generate tests for AI-generated code.
        * A good argument for using TDD: Generate tests first, based on prompts, and then generate the implementation.
* Licensing
    * I'm still looking for answers to the following questions.
    * Free access: Can people with IntelliJ IDEA Community Edition use AI Assistant and Junie? If yes, what are the limitations?
    * Access for Ultimate users
    * Is there additional licensing, and how does it work?
    * Are there token quotas?
* The "thank you" slide.


## Notes about the plan

After we decide on the contents, I need to go through the positioning document and think about how to include the slogans into the talk.

The following features are not currently in the plan due to time constraints but we can discuss if we should make changes to the plan to accommodate them.



* AI Assistant: Generate documentation
* AI Assistant: Explain errors in the runtime popup window
* AI Assistant: Explain the current branching situation in git (helpful for conflict resolution)
* AI Chat : /commands and #mentions
* Junie code example: Let it add a library and use it (e.g. use upickle to save the tree elements to json)
    * It can be either as a video or a quick live coding example
* A short discussion of MCP
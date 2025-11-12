"""
Direct Flow: User → LLM Tool Call → Tool Result → Final Assistant Message
Implements the exact 4-step flow requested
"""

import os
import asyncio
import json
from typing import Any, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BlueskyFlowAgent:
    """Agent that implements the exact 4-step flow"""
    
    def __init__(self):
        self.llm = None
        self.mcp_client = None
        self.available_tools = {}
        self._stdio_context = None
        self._session_context = None
    
    async def setup(self):
        """Setup LLM and MCP connection"""
        # Initialize LLM
        self.llm = ChatOpenAI(
            base_url="https://llms-inference.innkube.fim.uni-passau.de",
            api_key=os.getenv("UNIVERSITY_LLM_API_KEY"),
            model="llama3.1",
            temperature=0.0
        )
        
        # Setup MCP connection
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "server.py"],
            env={
                "BLUESKY_IDENTIFIER": os.getenv("BLUESKY_IDENTIFIER", ""),
                "BLUESKY_APP_PASSWORD": os.getenv("BLUESKY_APP_PASSWORD", ""),
            }
        )
        
        # Create persistent connection
        self._stdio_context = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()
        
        self._session_context = ClientSession(read_stream, write_stream)
        self.mcp_client = await self._session_context.__aenter__()
        
        await self.mcp_client.initialize()
        
        # Load available tools
        tools_result = await self.mcp_client.list_tools()
        for tool in tools_result.tools:
            self.available_tools[tool.name] = {
                'description': tool.description,
                'schema': tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            }
        
        print(f"✅ Setup complete with {len(self.available_tools)} tools")
    
    def get_tools_description(self) -> str:
        """Get formatted description of all available tools"""
        tools_desc = "Available Bluesky tools:\n"
        for name, info in self.available_tools.items():
            tools_desc += f"- {name}: {info['description']}\n"
        return tools_desc
    
    async def step1_user_message(self, user_input: str) -> str:
        """Step 1: Process user message"""
        print(f"📝 STEP 1 - User Message: {user_input}")
        return user_input
    
    async def step2_llm_tool_call(self, user_input: str) -> Dict[str, Any]:
        """Step 2: LLM decides which tool to call and with what parameters"""
        print("🤖 STEP 2 - LLM generating tool call...")
        
        system_prompt = f"""You are a Bluesky social media assistant. 

        {self.get_tools_description()}

        The user will ask you to do something. You need to decide which tool to call and what parameters to use.

        IMPORTANT: Respond with ONLY a valid JSON object. Do not wrap it in markdown code blocks or add any other text.

        Respond with a JSON object containing:
        - "tool_name": the name of the tool to call
        - "parameters": object with the parameters for the tool
        - "reasoning": brief explanation of why you chose this tool

        If no tool is needed, set "tool_name" to null.

        Examples:
        - If user asks "check my auth status" → {{"tool_name": "check_auth_status", "parameters": {{}}, "reasoning": "User wants to check authentication"}}
        - If user asks "get my profile" → {{"tool_name": "get_profile", "parameters": {{}}, "reasoning": "User wants their profile info"}}
        - If user asks "post hello world" → {{"tool_name": "send_post", "parameters": {{"text": "hello world"}}, "reasoning": "User wants to create a post"}}

        Remember: Return ONLY the JSON object, nothing else."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Clean the response content - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            # Parse the JSON response
            tool_call = json.loads(content)
            print(f"🔧 Tool Call: {json.dumps(tool_call, indent=2)}")
            return tool_call
        except json.JSONDecodeError:
            print(f"❌ Failed to parse LLM response as JSON: {response.content}")
            return {"tool_name": None, "parameters": {}, "reasoning": "Failed to parse response"}
    
    async def step3_tool_execution(self, tool_call: Dict[str, Any]) -> str:
        """Step 3: Execute the tool and get result"""
        print("⚡ STEP 3 - Executing tool...")
        
        tool_name = tool_call.get("tool_name")
        parameters = tool_call.get("parameters", {})
        
        if not tool_name:
            return "No tool was called"
        
        if tool_name not in self.available_tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            # Execute the MCP tool
            result = await self.mcp_client.call_tool(tool_name, parameters)
            
            # Handle the result
            if hasattr(result, 'content') and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    tool_result = content.text
                else:
                    tool_result = str(content)
            else:
                tool_result = str(result)
            
            print(f"📤 Tool Result: {tool_result[:2000]}..." if len(tool_result) > 2000 else f"📤 Tool Result: {tool_result}")
            return tool_result
            
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    async def step4_final_response(self, user_input: str, tool_call: Dict[str, Any], tool_result: str) -> str:
        """Step 4: Generate final assistant message based on tool result"""
        print("💬 STEP 4 - Generating final assistant message...")
        
        system_prompt = """You are a helpful Bluesky social media assistant. 

        The user asked you to do something, you called a tool, and got a result. 
        Now provide a friendly, helpful response to the user explaining what happened and the result.

        Be conversational and helpful. Explain the result in plain language."""
                
        context_prompt = f"""User asked: {user_input}

        I called the tool: {tool_call.get('tool_name', 'none')}
        With parameters: {json.dumps(tool_call.get('parameters', {}), indent=2)}
        Reasoning: {tool_call.get('reasoning', 'none')}

        Tool result: {tool_result}

        Please provide a helpful response to the user explaining what happened and the result."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        final_message = response.content
        
        print(f"✨ Final Assistant Message: {final_message}")
        return final_message
    
    async def process_user_request(self, user_input: str) -> str:
        """Process the complete 4-step flow"""
        print("=" * 60)
        print("🚀 Starting 4-Step Flow")
        print("=" * 60)
        
        # Step 1: User message
        await self.step1_user_message(user_input)
        
        # Step 2: LLM generates tool call
        tool_call = await self.step2_llm_tool_call(user_input)
        
        # Step 3: Execute tool
        tool_result = await self.step3_tool_execution(tool_call)
        
        # Step 4: Final assistant message
        final_response = await self.step4_final_response(user_input, tool_call, tool_result)
        
        print("=" * 60)
        print("✅ Flow Complete")
        print("=" * 60)
        
        return final_response
    
    async def cleanup(self):
        """Clean up connections"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)


async def main():
    """Main function to demonstrate the 4-step flow"""
    print("🚀 Bluesky 4-Step Flow Agent")
    print("=" * 40)
    
    # Check credentials
    missing = []
    if not os.getenv("BLUESKY_IDENTIFIER"):
        missing.append("BLUESKY_IDENTIFIER")
    if not os.getenv("BLUESKY_APP_PASSWORD"):
        missing.append("BLUESKY_APP_PASSWORD")
    if not os.getenv("UNIVERSITY_LLM_API_KEY"):
        missing.append("UNIVERSITY_LLM_API_KEY")
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        return
    
    agent = BlueskyFlowAgent()
    
    try:
        await agent.setup()
        
        # Example requests to test the flow
        test_requests = [
            "Check my authentication status",
            "Get my profile information",
            "What's on my timeline?"
        ]
        
        print("\n🎯 Agent ready! Try these examples or enter your own:")
        for i, req in enumerate(test_requests, 1):
            print(f"   {i}. {req}")
        
        while True:
            user_input = input("\n💬 Your request (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(test_requests):
                    user_input = test_requests[idx]
                else:
                    print("Invalid example number")
                    continue
            
            if user_input:
                await agent.process_user_request(user_input)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await agent.cleanup()
        print("👋 Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main()) 
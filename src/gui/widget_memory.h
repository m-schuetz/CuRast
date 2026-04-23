

void makeMemory(){

	auto editor = CuRast::instance;

	if(CuRastSettings::showMemoryInfos){

		// auto windowSize = ImGui::GetWindowSize();
		ImVec2 windowSize = {800, 600};
		ImGui::SetNextWindowPos({
			(VKRenderer::width - windowSize.x) / 2, 
			(VKRenderer::height - windowSize.y) / 2, }, 
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

		static bool open = true;
		if(ImGui::Begin("Memory", &open)){

			static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;

			ImGui::Text("List of allocations made via CURuntime::alloc and allocVirtual.");
			ImGui::Text(" ");

			ImGui::Text("===============================");
			ImGui::Text("## CUDA MEMORY ALLOCATIONS");
			ImGui::Text("===============================");

			if(ImGui::BeginTable("Memory", 2, flags)){

				ImGui::TableSetupColumn("Label",             ImGuiTableColumnFlags_WidthStretch, 3.0f);
				ImGui::TableSetupColumn("Allocated Memory",  ImGuiTableColumnFlags_WidthStretch, 1.0f);

				ImGui::TableHeadersRow();

				unordered_map<string, int64_t> cummulative;

				int64_t sum = 0;
				for(auto allocation : MemoryManager::allocations){
					
					ImGui::TableNextRow();

					ImGui::TableNextColumn();
					ImGui::Text(allocation.label.c_str());

					ImGui::TableNextColumn();
					string strMemory = format(getSaneLocale(), "{:L}", allocation.size);
					alignRight(strMemory);
					ImGui::Text(strMemory.c_str());

					sum += allocation.size;
					cummulative[allocation.label] += allocation.size;
				}

				ImGui::EndTable();
			}

			ImGui::Text("===============================");
			ImGui::Text("## VIRTUAL CUDA MEMORY ALLOCATIONS");
			ImGui::Text("===============================");

			if(ImGui::BeginTable("Memory", 2, flags)){

				ImGui::TableSetupColumn("Label",             ImGuiTableColumnFlags_WidthStretch, 3.0f);
				ImGui::TableSetupColumn("allocated/comitted memory",  ImGuiTableColumnFlags_WidthStretch, 1.0f);

				ImGui::TableHeadersRow();

				int64_t sum = 0;
				for(CudaVirtualMemory* memory : MemoryManager::cudaVirtual){
					
					ImGui::TableNextRow();

					ImGui::TableNextColumn();
					ImGui::Text(memory->label.c_str());

					ImGui::TableNextColumn();
					string strMemory = format(getSaneLocale(), "{:L}", memory->comitted);
					alignRight(strMemory);
					ImGui::Text(strMemory.c_str());

					sum += memory->comitted;
				}

				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("-----------------------");
					ImGui::TableNextColumn();
					ImGui::Text(" ");

					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("Total");
					ImGui::TableNextColumn();
					string strTotal = format(getSaneLocale(), "{:L}", sum);
					alignRight(strTotal);
					ImGui::Text(strTotal.c_str());
				}

				ImGui::EndTable();
			}

			ImGui::Text("===============================");
			ImGui::Text("## Vulkan-CUDA SHARED MEMORY ALLOCATIONS");
			ImGui::Text("===============================");

			if(ImGui::BeginTable("Memory", 2, flags)){

				ImGui::TableSetupColumn("Label",             ImGuiTableColumnFlags_WidthStretch, 3.0f);
				ImGui::TableSetupColumn("allocated/comitted memory",  ImGuiTableColumnFlags_WidthStretch, 1.0f);

				ImGui::TableHeadersRow();

				int64_t sum = 0;
				for(VulkanCudaSharedMemory* memory : MemoryManager::vulkanCudaShared){
					
					ImGui::TableNextRow();

					ImGui::TableNextColumn();
					ImGui::Text(memory->label.c_str());

					ImGui::TableNextColumn();
					string strMemory = format(getSaneLocale(), "{:L}", memory->comitted);
					alignRight(strMemory);
					ImGui::Text(strMemory.c_str());

					sum += memory->comitted;
				}

				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("-----------------------");
					ImGui::TableNextColumn();
					ImGui::Text(" ");

					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("Total");
					ImGui::TableNextColumn();
					string strTotal = format(getSaneLocale(), "{:L}", sum);
					alignRight(strTotal);
					ImGui::Text(strTotal.c_str());
				}

				ImGui::EndTable();
			}

			ImGui::Text("===============================");
			ImGui::Text("## Vulkan Buffers");
			ImGui::Text("===============================");

			if(ImGui::BeginTable("Memory", 2, flags)){

				ImGui::TableSetupColumn("Label",             ImGuiTableColumnFlags_WidthStretch, 3.0f);
				ImGui::TableSetupColumn("allocated/comitted memory",  ImGuiTableColumnFlags_WidthStretch, 1.0f);

				ImGui::TableHeadersRow();

				int64_t sum = 0;
				for(VKBuffer* buffer : MemoryManager::allocations_vulkan){
					
					ImGui::TableNextRow();

					ImGui::TableNextColumn();
					ImGui::Text(buffer->label.c_str());

					ImGui::TableNextColumn();
					string strMemory = format(getSaneLocale(), "{:L}", buffer->size);
					alignRight(strMemory);
					ImGui::Text(strMemory.c_str());

					sum += buffer->size;
				}

				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("-----------------------");
					ImGui::TableNextColumn();
					ImGui::Text(" ");

					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::Text("Total");
					ImGui::TableNextColumn();
					string strTotal = format(getSaneLocale(), "{:L}", sum);
					alignRight(strTotal);
					ImGui::Text(strTotal.c_str());
				}

				ImGui::EndTable();
			}

		}

		CuRastSettings::showMemoryInfos = open;

		ImGui::End();
	}

}
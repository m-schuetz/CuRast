void makeKernels(){

	auto editor = CuRast::instance;

	if(CuRastSettings::showKernelInfos){

		ImVec2 kernelWindowSize = {1400, 600};
		ImGui::SetNextWindowPos({
			(VKRenderer::width - kernelWindowSize.x) / 2, 
			(VKRenderer::height - kernelWindowSize.y) / 2, }, 
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(kernelWindowSize, ImGuiCond_Once);

		bool open = CuRastSettings::showKernelInfos;
		if(ImGui::Begin("Kernels", &open)){

			static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;


			auto printKernels = [&](string label, CudaModularProgram* program){

				string strDefines = "";
				for(int i = 0; i < program->defines.size(); i++){
					strDefines += program->defines[i];

					if(i < program->defines.size() - 1){
						strDefines += ", ";
					}
				}
				

				string strlabel = format("## {}, {}", label, strDefines);
				ImGui::Text("===============================");
				ImGui::Text(strlabel.c_str());
				ImGui::Text("===============================");
				
				if(ImGui::BeginTable("Kernels##listOfKernels", 5, flags))
				{
					ImGui::TableSetupColumn("Name",       ImGuiTableColumnFlags_WidthStretch, 3.0f);
					ImGui::TableSetupColumn("registers",  ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("shared mem", ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("max threads/block", ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("blocks(64, 128, 256)/SM", ImGuiTableColumnFlags_WidthStretch, 1.0f);

					ImGui::TableHeadersRow();

					vector<string> kernelNames;
					for(auto [name, function] : program->kernels){
						if(name.contains("memcpy")) continue;
						if(name.contains("memset")) continue;
						kernelNames.push_back(name);
					}
					sort(kernelNames.begin(), kernelNames.end());

					// for(auto [name, function] : program->kernels){
					for(string name : kernelNames){

						auto function = program->kernels[name];

						// if(name.contains("memcpy")) continue;
						// if(name.contains("memset")) continue;

						ImGui::TableNextRow();
				
						int maxThreadsPerBlock = 0;
						int registersPerThread;
						int sharedMemory;
						cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
						cuFuncGetAttribute(&registersPerThread, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
						cuFuncGetAttribute(&sharedMemory, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function);

						int numBlocksPerSM64;
						int numBlocksPerSM128;
						int numBlocksPerSM256;
						cuOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocksPerSM64, function, 64, 0);
						cuOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocksPerSM128, function, 128, 0);
						cuOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocksPerSM256, function, 256, 0);

						string strThreadsPerBlock = format("{}", maxThreadsPerBlock);
						if(maxThreadsPerBlock == 0) strThreadsPerBlock = "?";
						string strRegisters = format("{}", registersPerThread);
						string strSharedMem = format(getSaneLocale(), "{:L}", sharedMemory);
						string strBlocksPerSM = format(getSaneLocale(), "{:3L}, {:3L}, {:3L}", numBlocksPerSM64, numBlocksPerSM128, numBlocksPerSM256);

						ImGui::TableNextColumn();
						ImGui::Text(name.c_str());

						ImGui::TableNextColumn();
						alignRight(strRegisters);
						ImGui::Text(strRegisters.c_str());

						ImGui::TableNextColumn();
						alignRight(strSharedMem);
						ImGui::Text(strSharedMem.c_str());

						ImGui::TableNextColumn();
						alignRight(strThreadsPerBlock);
						ImGui::Text(strThreadsPerBlock.c_str());

						ImGui::TableNextColumn();
						alignRight(strBlocksPerSM);
						ImGui::Text(strBlocksPerSM.c_str());
					}

					ImGui::EndTable();
				}
			};

			for (CudaModularProgram* prog : CudaModularProgram::instances) {
				
				printKernels(prog->modules[0]->name, prog);
			}
		}

		// settings.showKernelInfos = open;

		ImGui::End();
	}

}
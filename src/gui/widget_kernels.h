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


			// group kernels by the .cu file they're defined in
			map<string, vector<KernelInfo>> modules;
			for(KernelInfo& info : getKernelInfos()){
				modules[info.module].push_back(info);
			}

			for(auto& [module, kernels] : modules){

				string strlabel = format("## {}", module);
				ImGui::Text("===============================");
				ImGui::Text(strlabel.c_str());
				ImGui::Text("===============================");

				ImGui::PushID(module.c_str());
				if(ImGui::BeginTable("Kernels##listOfKernels", 5, flags))
				{
					ImGui::TableSetupColumn("Name",       ImGuiTableColumnFlags_WidthStretch, 3.0f);
					ImGui::TableSetupColumn("registers",  ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("shared mem", ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("max threads/block", ImGuiTableColumnFlags_WidthStretch, 1.0f);
					ImGui::TableSetupColumn("blocks(64, 128, 256)/SM", ImGuiTableColumnFlags_WidthStretch, 1.0f);

					ImGui::TableHeadersRow();

					sort(kernels.begin(), kernels.end(), [](const KernelInfo& a, const KernelInfo& b){ return a.name < b.name; });

					for(KernelInfo& info : kernels){

						ImGui::TableNextRow();

						cudaFuncAttributes attributes = {};
						cudaFuncGetAttributes(&attributes, info.function);

						int numBlocksPerSM64;
						int numBlocksPerSM128;
						int numBlocksPerSM256;
						cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM64, info.function, 64, 0);
						cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM128, info.function, 128, 0);
						cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM256, info.function, 256, 0);

						string strThreadsPerBlock = format("{}", attributes.maxThreadsPerBlock);
						if(attributes.maxThreadsPerBlock == 0) strThreadsPerBlock = "?";
						string strRegisters = format("{}", attributes.numRegs);
						string strSharedMem = format(getSaneLocale(), "{:L}", attributes.sharedSizeBytes);
						string strBlocksPerSM = format(getSaneLocale(), "{:3L}, {:3L}, {:3L}", numBlocksPerSM64, numBlocksPerSM128, numBlocksPerSM256);

						ImGui::TableNextColumn();
						ImGui::Text(info.name.c_str());

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
				ImGui::PopID();
			}
		}

		// settings.showKernelInfos = open;

		ImGui::End();
	}

}